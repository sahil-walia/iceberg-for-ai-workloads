"""EXP-9: Write-Audit-Publish (WAP) round-trip latency benchmark.

Measures the operational overhead of Iceberg's Write-Audit-Publish pattern
relative to a direct-write baseline. WAP is a native Iceberg/Spark capability
that stages a snapshot before committing, enabling a quality-gate audit step
before the data becomes visible to readers.

For each dataset size the experiment runs:
  - Baseline: direct append, no WAP (measures raw write cost)
  - WAP stage: write with wap.id set (creates staged, invisible snapshot)
  - WAP audit: full-table aggregate scan against the staged snapshot
  - WAP publish: cherrypick_snapshot to promote staged snapshot to current
  - WAP total: stage + audit + publish

Key claim under test (Pattern 3, paper Section 5.3): WAP overhead is dominated
by the audit scan, not by the commit mechanics. The publish step is a metadata
operation and should be near-zero relative to the write cost.

All operations run on a local filesystem warehouse. No object-store credentials
required. The experiment isolates WAP mechanics, not I/O throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

ICEBERG_VERSION = "1.11.0"
SPARK_VERSION = "4.0"
SPARK_PACKAGE = (
    f"org.apache.iceberg:iceberg-spark-runtime-{SPARK_VERSION}_2.13:{ICEBERG_VERSION}"
)
CATALOG = "local"
NAMESPACE = "benchmark"
TABLE = "wap_dataset"
FULL_TABLE = f"{CATALOG}.{NAMESPACE}.{TABLE}"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Dataset sizes to test (rows)
DEFAULT_SIZES = [100_000, 500_000, 1_000_000, 5_000_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="Row counts to test (default: 100000 500000 1000000 5000000)",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--shuffle-partitions", type=int, default=4)
    return parser.parse_args()


def configure_java_17() -> None:
    def java_major() -> int | None:
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        match = re.search(r'version "(\d+)', result.stderr or result.stdout)
        return int(match.group(1)) if match else None

    if 17 <= (java_major() or 0) < 25:
        return
    homebrew_java_17 = Path(
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    )
    if homebrew_java_17.exists():
        java_home = str(homebrew_java_17)
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
    if not (17 <= (java_major() or 0) < 25):
        raise RuntimeError(
            "EXP-9 requires Java 17–24. Install Java 17 and set JAVA_HOME."
        )


def build_spark(warehouse: Path, shuffle_partitions: int) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.master("local[*]")
        .appName("IEEEBigData2026-WAP-Latency")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.inMemoryColumnarStorage.enableVectorizedReader", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
        .config("spark.jars.packages", SPARK_PACKAGE)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", str(warehouse))
        .getOrCreate()
    )


def make_dataframe(spark: SparkSession, n_rows: int):
    return spark.range(n_rows).selectExpr(
        "format_string('user_%06d', id % 100000) AS entity_id",
        "CAST(id AS BIGINT) AS event_sequence",
        "CAST((id * 17 + 11) % 1000003 AS BIGINT) AS feature_1",
        "CAST((id * 31 + 7) % 1000033 AS BIGINT) AS feature_2",
        "element_at(array('A','B','C','D'), CAST((id % 4) + 1 AS INT)) AS feature_group",
    )


def reset_table(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{NAMESPACE}")
    spark.sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")
    spark.sql(
        f"""
        CREATE TABLE {FULL_TABLE} (
            entity_id STRING NOT NULL,
            event_sequence BIGINT NOT NULL,
            feature_1 BIGINT NOT NULL,
            feature_2 BIGINT NOT NULL,
            feature_group STRING NOT NULL
        )
        USING iceberg
        TBLPROPERTIES (
            'format-version' = '2',
            'write.parquet.compression-codec' = 'zstd',
            'write.target-file-size-bytes' = '134217728',
            'write.wap.enabled' = 'true'
        )
        """
    )


def get_current_snapshot_id(spark: SparkSession) -> int | None:
    row = spark.sql(
        f"SELECT snapshot_id FROM {FULL_TABLE}.snapshots "
        f"ORDER BY committed_at DESC LIMIT 1"
    ).first()
    return int(row["snapshot_id"]) if row else None


def get_wap_snapshot_id(spark: SparkSession, wap_id: str) -> int:
    row = spark.sql(
        f"SELECT snapshot_id FROM {FULL_TABLE}.snapshots "
        f"WHERE summary['wap.id'] = '{wap_id}' "
        f"ORDER BY committed_at DESC LIMIT 1"
    ).first()
    if row is None:
        raise RuntimeError(f"No WAP snapshot found for wap.id={wap_id!r}")
    return int(row["snapshot_id"])


def aggregate_checksum(spark: SparkSession, snapshot_id: int | None = None) -> dict[str, int]:
    if snapshot_id is not None:
        source = f"{FULL_TABLE} VERSION AS OF {snapshot_id}"
    else:
        source = FULL_TABLE
    spark.catalog.clearCache()
    row = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count,
               SUM(feature_1) AS checksum_1,
               SUM(feature_2) AS checksum_2
        FROM {source}
        """
    ).first()
    return {k: int(row[k]) for k in row.asDict()}


def time_operation(fn) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def measure_direct_write(spark: SparkSession, n_rows: int, repetitions: int, warmups: int) -> dict:
    """Baseline: direct append with no WAP staging."""
    timings: list[float] = []
    checksums: list[dict] = []
    for i in range(warmups + repetitions):
        reset_table(spark)
        # Disable WAP for this measurement
        spark.sql(
            f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES ('write.wap.enabled' = 'false')"
        )
        df = make_dataframe(spark, n_rows)
        elapsed, _ = time_operation(lambda: df.coalesce(1).writeTo(FULL_TABLE).append())
        if i >= warmups:
            timings.append(elapsed)
            checksums.append(aggregate_checksum(spark))
    return {"timings_s": timings, "checksums": checksums}


def measure_wap_phases(spark: SparkSession, n_rows: int, repetitions: int, warmups: int) -> dict:
    """WAP workflow: stage → audit → publish."""
    stage_timings: list[float] = []
    audit_timings: list[float] = []
    publish_timings: list[float] = []
    checksums_staged: list[dict] = []
    checksums_published: list[dict] = []

    for i in range(warmups + repetitions):
        reset_table(spark)
        wap_id = f"wap-run-{i}"

        # Stage
        spark.conf.set("spark.wap.id", wap_id)
        df = make_dataframe(spark, n_rows)
        stage_elapsed, _ = time_operation(lambda: df.coalesce(1).writeTo(FULL_TABLE).append())
        spark.conf.unset("spark.wap.id")

        wap_snapshot = get_wap_snapshot_id(spark, wap_id)

        # Audit: full aggregate scan against the staged snapshot
        spark.catalog.clearCache()
        audit_elapsed, staged_checksums = time_operation(
            lambda: aggregate_checksum(spark, wap_snapshot)
        )

        # Publish: cherrypick the staged snapshot to current
        publish_elapsed, _ = time_operation(
            lambda: spark.sql(
                f"CALL {CATALOG}.system.cherrypick_snapshot("
                f"table => '{NAMESPACE}.{TABLE}', "
                f"snapshot_id => {wap_snapshot})"
            ).first()
        )

        # Verify published content
        spark.catalog.clearCache()
        published_checksums = aggregate_checksum(spark)

        if i >= warmups:
            stage_timings.append(stage_elapsed)
            audit_timings.append(audit_elapsed)
            publish_timings.append(publish_elapsed)
            checksums_staged.append(staged_checksums)
            checksums_published.append(published_checksums)

        # Validate checksums match
        if staged_checksums != published_checksums:
            raise RuntimeError(
                f"WAP checksum mismatch: staged={staged_checksums} published={published_checksums}"
            )

    return {
        "stage_timings_s": stage_timings,
        "audit_timings_s": audit_timings,
        "publish_timings_s": publish_timings,
        "checksums_staged": checksums_staged,
        "checksums_published": checksums_published,
    }


def summarize(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def system_metadata(spark: SparkSession) -> dict[str, Any]:
    try:
        memory_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    except Exception:
        memory_bytes = None
    jvm = spark.sparkContext._jvm
    return {
        "backend": "local-filesystem",
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python_version": platform.python_version(),
        "spark_version": spark.version,
        "iceberg_version": ICEBERG_VERSION,
        "java_version": jvm.java.lang.System.getProperty("java.version"),
    }


def write_outputs(summary: dict[str, Any], timestamp: str) -> list[Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"iceberg_wap_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for size_result in summary["size_results"]:
        n = size_result["n_rows"]
        for phase in ("direct_write", "wap_stage", "wap_audit", "wap_publish", "wap_total"):
            for rep, t in enumerate(size_result["raw_timings"].get(phase + "_s", []), start=1):
                rows.append({"n_rows": n, "phase": phase, "repetition": rep, "time_s": t})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n_rows", "phase", "repetition", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["n_rows"] for r in summary["size_results"]]
    direct = [r["summary"]["direct_write"]["median_s"] for r in summary["size_results"]]
    total_wap = [r["summary"]["wap_total"]["median_s"] for r in summary["size_results"]]
    audit = [r["summary"]["wap_audit"]["median_s"] for r in summary["size_results"]]
    publish = [r["summary"]["wap_publish"]["median_s"] for r in summary["size_results"]]

    x = range(len(sizes))
    labels = [f"{n // 1_000}K" if n < 1_000_000 else f"{n // 1_000_000}M" for n in sizes]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(labels, direct, marker="o", label="Direct write (baseline)")
    ax.plot(labels, total_wap, marker="s", label="WAP total (stage+audit+publish)")
    ax.plot(labels, audit, marker="^", linestyle="--", label="WAP audit scan only")
    ax.plot(labels, publish, marker="x", linestyle=":", label="WAP publish (cherrypick)")
    ax.set_xlabel("Dataset size (rows)")
    ax.set_ylabel("Median latency (seconds)")
    ax.set_title("WAP overhead vs. direct write by dataset size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    return [json_path, csv_path, svg_path, pdf_path]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    configure_java_17()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with tempfile.TemporaryDirectory(prefix="iceberg-exp9-") as tmp:
        warehouse = Path(tmp) / "warehouse"
        spark = build_spark(warehouse, args.shuffle_partitions)
        spark.sparkContext.setLogLevel("WARN")

        environment = system_metadata(spark)
        size_results: list[dict] = []

        try:
            for n_rows in args.sizes:
                print(f"\n--- n_rows={n_rows:,} ---")

                # Direct write baseline
                print(f"  Direct write ({args.repetitions} reps)...")
                direct = measure_direct_write(spark, n_rows, args.repetitions, args.warmups)
                direct_summary = summarize(direct["timings_s"])
                print(f"    median={direct_summary['median_s']:.3f}s")

                # WAP phases
                print(f"  WAP (stage+audit+publish, {args.repetitions} reps)...")
                wap = measure_wap_phases(spark, n_rows, args.repetitions, args.warmups)
                total_timings = [
                    s + a + p
                    for s, a, p in zip(
                        wap["stage_timings_s"],
                        wap["audit_timings_s"],
                        wap["publish_timings_s"],
                    )
                ]
                stage_summary = summarize(wap["stage_timings_s"])
                audit_summary = summarize(wap["audit_timings_s"])
                publish_summary = summarize(wap["publish_timings_s"])
                total_summary = summarize(total_timings)

                overhead_ratio = (
                    total_summary["median_s"] / direct_summary["median_s"]
                    if direct_summary["median_s"] > 0 else None
                )

                print(f"    stage={stage_summary['median_s']:.3f}s  "
                      f"audit={audit_summary['median_s']:.3f}s  "
                      f"publish={publish_summary['median_s']:.4f}s  "
                      f"total={total_summary['median_s']:.3f}s  "
                      f"overhead={overhead_ratio:.2f}x")

                size_results.append({
                    "n_rows": n_rows,
                    "summary": {
                        "direct_write": direct_summary,
                        "wap_stage": stage_summary,
                        "wap_audit": audit_summary,
                        "wap_publish": publish_summary,
                        "wap_total": total_summary,
                        "overhead_ratio": overhead_ratio,
                    },
                    "raw_timings": {
                        "direct_write_s": direct["timings_s"],
                        "wap_stage_s": wap["stage_timings_s"],
                        "wap_audit_s": wap["audit_timings_s"],
                        "wap_publish_s": wap["publish_timings_s"],
                        "wap_total_s": total_timings,
                    },
                    "content_verified": True,
                })

        finally:
            spark.stop()

    summary = {
        "experiment": "EXP-9: WAP round-trip latency vs. direct write",
        "timestamp_utc": timestamp,
        "scope": (
            "Local-filesystem microbenchmark. Measures WAP stage/audit/publish "
            "latency relative to a direct-append baseline. Validates Pattern 3 "
            "(Write-Audit-Publish for LLM fine-tuning data curation)."
        ),
        "parameters": {
            "sizes": args.sizes,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "shuffle_partitions": args.shuffle_partitions,
            "format_version": 2,
        },
        "environment": environment,
        "size_results": size_results,
    }
    paths = write_outputs(summary, timestamp)
    print("\n=== EXP-9 Complete ===")
    print(f"{'Size':>8}  {'Direct':>8}  {'Stage':>8}  {'Audit':>8}  {'Publish':>9}  {'Total':>8}  {'Overhead':>9}")
    for r in size_results:
        n = r["n_rows"]
        s = r["summary"]
        label = f"{n // 1_000}K" if n < 1_000_000 else f"{n // 1_000_000}M"
        print(
            f"{label:>8}  "
            f"{s['direct_write']['median_s']:>8.3f}s  "
            f"{s['wap_stage']['median_s']:>8.3f}s  "
            f"{s['wap_audit']['median_s']:>8.3f}s  "
            f"{s['wap_publish']['median_s']:>9.4f}s  "
            f"{s['wap_total']['median_s']:>8.3f}s  "
            f"{s['overhead_ratio']:>8.2f}x"
        )
    for p in paths:
        print(f"Saved {p}")
    return summary


if __name__ == "__main__":
    run_experiment(parse_args())
