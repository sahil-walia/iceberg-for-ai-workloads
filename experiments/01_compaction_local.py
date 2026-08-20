"""EXP-7: controlled Apache Iceberg small-file compaction benchmark.

The benchmark uses Spark's Iceberg integration for every table operation. It
creates one Iceberg commit per input batch, measures repeated aggregate scans
without explicit Spark caching,
calls Iceberg's ``rewrite_data_files`` procedure, and verifies that compaction
preserves the row count and deterministic checksums.

This is a local-filesystem microbenchmark. It demonstrates the direction and
local magnitude of small-file overhead; it does not emulate object-store
latency or reproduce the scale of the cited AWS deployment.
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
SPARK_PACKAGE = (
    f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION}"
)
CATALOG = "local"
NAMESPACE = "benchmark"
TABLE = "feature_store"
FULL_TABLE = f"{CATALOG}.{NAMESPACE}.{TABLE}"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--rows-per-batch", type=int, default=2_000)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--shuffle-partitions", type=int, default=4)
    parser.add_argument(
        "--warehouse",
        type=Path,
        help=(
            "Optional persistent warehouse path; defaults to a temporary "
            "directory."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batches",
        "rows_per_batch",
        "measure_every",
        "repetitions",
        "shuffle_partitions",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")


def configure_java_17() -> None:
    """Select an installed Java 17+ runtime before Spark starts."""

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

    if (java_major() or 0) >= 17:
        if (java_major() or 0) < 25:
            return
    homebrew_java_17 = Path(
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    )
    if homebrew_java_17.exists():
        java_home = str(homebrew_java_17)
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
    else:
        java_home_tool = Path("/usr/libexec/java_home")
        if not java_home_tool.exists():
            java_home_tool = None
    major = java_major() or 0
    if not (17 <= major < 25) and java_home_tool:
        result = subprocess.run(
            [str(java_home_tool), "-v", "17"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
    major = java_major() or 0
    if not (17 <= major < 25):
        raise RuntimeError(
            "EXP-7 requires Java 17 through 24 (Java 17 recommended). "
            "Install Java 17 and set JAVA_HOME before running it."
        )


def build_spark(warehouse: Path, shuffle_partitions: int) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.master("local[*]")
        .appName("IEEEBigData2026-Iceberg-Compaction")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "false")
        .config(
            "spark.sql.inMemoryColumnarStorage.enableVectorizedReader",
            "false",
        )
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config("spark.jars.packages", SPARK_PACKAGE)
        .config(
            "spark.sql.extensions",
            (
                "org.apache.iceberg.spark.extensions."
                "IcebergSparkSessionExtensions"
            ),
        )
        .config(
            f"spark.sql.catalog.{CATALOG}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", str(warehouse))
        .getOrCreate()
    )


def create_table(spark: SparkSession) -> None:
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
            'write.target-file-size-bytes' = '134217728'
        )
        """
    )


def append_batch(
    spark: SparkSession, batch_index: int, rows_per_batch: int
) -> None:
    offset = batch_index * rows_per_batch
    batch = spark.range(rows_per_batch).selectExpr(
        f"format_string('user_%05d', ({offset} + id) % 10000) AS entity_id",
        f"CAST({offset} + id AS BIGINT) AS event_sequence",
        f"CAST((({offset} + id) * 17 + 11) % 1000003 AS BIGINT) AS feature_1",
        f"CAST((({offset} + id) * 31 + 7) % 1000033 AS BIGINT) AS feature_2",
        (
            "element_at(array('A','B','C','D'), "
            f"CAST((({offset} + id) % 4) + 1 AS INT)) AS feature_group"
        ),
    )
    batch.coalesce(1).writeTo(FULL_TABLE).append()


def table_state(spark: SparkSession) -> dict[str, int]:
    files = spark.sql(
        f"""
        SELECT COUNT(*) AS file_count,
               COALESCE(SUM(record_count), 0) AS metadata_row_count
        FROM {FULL_TABLE}.files
        """
    ).first()
    manifests = spark.sql(
        f"SELECT COUNT(*) AS manifest_count FROM {FULL_TABLE}.manifests"
    ).first()
    snapshots = spark.sql(
        f"SELECT COUNT(*) AS snapshot_count FROM {FULL_TABLE}.snapshots"
    ).first()
    return {
        "file_count": int(files["file_count"]),
        "metadata_row_count": int(files["metadata_row_count"]),
        "manifest_count": int(manifests["manifest_count"]),
        "snapshot_count": int(snapshots["snapshot_count"]),
    }


def scan_once(spark: SparkSession) -> tuple[float, dict[str, int]]:
    spark.catalog.clearCache()
    started = time.perf_counter()
    row = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count,
               SUM(feature_1) AS checksum_1,
               SUM(feature_2) AS checksum_2,
               SUM(LENGTH(entity_id)) AS checksum_entity
        FROM {FULL_TABLE}
        """
    ).first()
    elapsed = time.perf_counter() - started
    checksums = {name: int(row[name]) for name in row.asDict()}
    return elapsed, checksums


def measure_scans(
    spark: SparkSession, warmups: int, repetitions: int
) -> tuple[list[float], dict[str, int]]:
    checksum: dict[str, int] | None = None
    for _ in range(warmups):
        _, checksum = scan_once(spark)
    timings: list[float] = []
    for _ in range(repetitions):
        elapsed, current = scan_once(spark)
        if checksum is not None and current != checksum:
            raise RuntimeError("Checksums changed between repeated scans")
        checksum = current
        timings.append(elapsed)
    assert checksum is not None
    return timings, checksum


def summarize_timings(timings: list[float]) -> dict[str, float]:
    ordered = sorted(timings)
    return {
        "median_s": statistics.median(ordered),
        "min_s": min(ordered),
        "max_s": max(ordered),
        "stdev_s": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def system_metadata(spark: SparkSession) -> dict[str, Any]:
    try:
        memory_bytes = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        memory_bytes = None
    jvm = spark.sparkContext._jvm
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python_version": platform.python_version(),
        "spark_version": spark.version,
        "iceberg_version": ICEBERG_VERSION,
        "iceberg_maven_package": SPARK_PACKAGE,
        "java_version": jvm.java.lang.System.getProperty("java.version"),
        "java_vendor": jvm.java.lang.System.getProperty("java.vendor"),
    }


def write_outputs(summary: dict[str, Any], timestamp: str) -> list[Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"iceberg_compaction_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phase",
                "batch",
                "file_count",
                "metadata_row_count",
                "manifest_count",
                "snapshot_count",
                "row_count",
                "repetition",
                "scan_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["raw_measurements"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoints = summary["checkpoints"]
    pre = [
        point for point in checkpoints
        if point["phase"] == "pre_compaction"
    ]
    post = next(
        point for point in checkpoints
        if point["phase"] == "post_compaction"
    )
    figure, axis = plt.subplots(figsize=(6.8, 3.6))
    axis.plot(
        [point["file_count"] for point in pre],
        [point["timing"]["median_s"] for point in pre],
        marker="o",
        label="Before compaction",
    )
    axis.scatter(
        [post["file_count"]],
        [post["timing"]["median_s"]],
        marker="s",
        s=55,
        label="After compaction",
    )
    axis.set_xlabel("Iceberg data files (count)")
    axis.set_ylabel("Median full-scan latency (seconds)")
    axis.set_title(
        "Local Iceberg scan latency before and after data-file rewrite"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(svg_path, format="svg")
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)
    return [json_path, csv_path, svg_path, pdf_path]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    configure_java_17()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temporary = None
    if args.warehouse:
        warehouse = args.warehouse.resolve()
        warehouse.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="iceberg-exp7-")
        warehouse = Path(temporary.name) / "warehouse"

    spark = build_spark(warehouse, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    raw_measurements: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    try:
        create_table(spark)
        environment = system_metadata(spark)
        print(
            f"Writing {args.batches} Iceberg commits "
            f"({args.rows_per_batch:,} rows each)"
        )
        last_checksum: dict[str, int] | None = None
        for batch_index in range(args.batches):
            append_batch(spark, batch_index, args.rows_per_batch)
            batch_number = batch_index + 1
            if (
                batch_number % args.measure_every == 0
                or batch_number == args.batches
            ):
                state = table_state(spark)
                timings, last_checksum = measure_scans(
                    spark, args.warmups, args.repetitions
                )
                timing_summary = summarize_timings(timings)
                point = {
                    "phase": "pre_compaction",
                    "batch": batch_number,
                    **state,
                    "timing": timing_summary,
                }
                checkpoints.append(point)
                for repetition, elapsed in enumerate(timings, start=1):
                    raw_measurements.append(
                        {
                            "phase": "pre_compaction",
                            "batch": batch_number,
                            **state,
                            "row_count": last_checksum["row_count"],
                            "repetition": repetition,
                            "scan_time_s": elapsed,
                        }
                    )
                print(
                    f"  {batch_number:4d} commits | "
                    f"{state['file_count']:4d} files | "
                    f"median scan {timing_summary['median_s']:.3f}s"
                )

        assert last_checksum is not None
        pre_state = table_state(spark)
        pre_timing = checkpoints[-1]["timing"]

        print("Calling Iceberg system.rewrite_data_files...")
        compaction_started = time.perf_counter()
        rewrite_result = spark.sql(
            f"""
            CALL {CATALOG}.system.rewrite_data_files(
                table => '{NAMESPACE}.{TABLE}',
                options => map(
                    'rewrite-all', 'true',
                    'target-file-size-bytes', '134217728'
                )
            )
            """
        ).first()
        compaction_duration = time.perf_counter() - compaction_started

        post_state = table_state(spark)
        post_timings, post_checksum = measure_scans(
            spark, args.warmups, args.repetitions
        )
        if post_checksum != last_checksum:
            raise RuntimeError(
                "Compaction changed table contents: "
                f"{last_checksum} != {post_checksum}"
            )
        post_timing = summarize_timings(post_timings)
        checkpoints.append(
            {
                "phase": "post_compaction",
                "batch": args.batches,
                **post_state,
                "timing": post_timing,
            }
        )
        for repetition, elapsed in enumerate(post_timings, start=1):
            raw_measurements.append(
                {
                    "phase": "post_compaction",
                    "batch": args.batches,
                    **post_state,
                    "row_count": post_checksum["row_count"],
                    "repetition": repetition,
                    "scan_time_s": elapsed,
                }
            )

        speedup = pre_timing["median_s"] / post_timing["median_s"]
        file_reduction = (
            100.0 * (pre_state["file_count"] - post_state["file_count"])
            / pre_state["file_count"]
        )
        summary = {
            "experiment": "EXP-7: Apache Iceberg small-file compaction",
            "timestamp_utc": timestamp,
            "scope": (
                "Local-filesystem microbenchmark; not an object-store or "
                "production-scale benchmark."
            ),
            "parameters": {
                "batches": args.batches,
                "rows_per_batch": args.rows_per_batch,
                "measure_every": args.measure_every,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "shuffle_partitions": args.shuffle_partitions,
                "format_version": 2,
            },
            "environment": environment,
            "rewrite_result": rewrite_result.asDict(recursive=True),
            "results": {
                "pre_compaction": {**pre_state, "timing": pre_timing},
                "post_compaction": {**post_state, "timing": post_timing},
                "compaction_duration_s": compaction_duration,
                "median_scan_speedup": speedup,
                "file_count_reduction_pct": file_reduction,
                "content_verified": True,
                "checksums": post_checksum,
            },
            "checkpoints": checkpoints,
            "raw_measurements": raw_measurements,
        }
        paths = write_outputs(summary, timestamp)
        print(
            f"Compaction: {pre_state['file_count']} -> "
            f"{post_state['file_count']} files "
            f"({file_reduction:.1f}% reduction)"
        )
        print(
            f"Median scan: {pre_timing['median_s']:.3f}s -> "
            f"{post_timing['median_s']:.3f}s ({speedup:.2f}x)"
        )
        print(f"Compaction duration: {compaction_duration:.3f}s")
        print("Content verification: PASS")
        for path in paths:
            print(f"Saved {path}")
        return summary
    finally:
        spark.stop()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    run_experiment(parse_args())
