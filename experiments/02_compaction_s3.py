"""EXP-12: Apache Iceberg small-file compaction benchmark on real Amazon S3.

Runs the same workload as EXP-7 (local filesystem) but against a real S3
bucket via the S3A client. This adds genuine object-store request latency
(per-object HTTP GET overhead, metadata API calls, and network round trips)
that is absent from the local-filesystem baseline and approximates from
below the production environment cited in the paper.

The expected speedup ratio is substantially higher than EXP-7's local result
because each data-file read in the pre-compaction scan requires a separate
S3 GET request (5–50 ms each at typical AWS network latency), whereas the
post-compaction single-file scan requires one GET.

Prerequisites:
  - AWS credentials configured and valid (run: aws sso login --profile Contributor-087354435437)
  - S3 bucket accessible (default: swalia-iceberg-benchmark, us-west-2)

Set AWS_PROFILE=Contributor-087354435437 before running, or pass --s3-bucket
and ensure AWS_DEFAULT_PROFILE / environment credentials are set.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

ICEBERG_VERSION = "1.11.0"
HADOOP_AWS_VERSION = "3.4.1"
AWS_SDK_VERSION = "1.12.261"

SPARK_PACKAGE = ",".join([
    f"org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:{ICEBERG_VERSION}",
    f"org.apache.hadoop:hadoop-aws:{HADOOP_AWS_VERSION}",
    f"com.amazonaws:aws-java-sdk-bundle:{AWS_SDK_VERSION}",
])

DEFAULT_S3_BUCKET = "swalia-iceberg-benchmark"
DEFAULT_S3_PREFIX = "iceberg-exp12"
CATALOG = "s3bench"
NAMESPACE = "benchmark"
TABLE = "feature_store"
FULL_TABLE = f"{CATALOG}.{NAMESPACE}.{TABLE}"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows-per-batch", type=int, default=5_000)
    parser.add_argument("--measure-every", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--shuffle-partitions", type=int, default=4)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument("--aws-profile", default="Contributor-087354435437")
    return parser.parse_args()


def configure_java_17() -> None:
    def java_major() -> int | None:
        try:
            result = subprocess.run(
                ["java", "-version"], capture_output=True, text=True, check=False
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
        raise RuntimeError("Requires Java 17–24.")


def resolve_aws_credentials(profile: str) -> tuple[str, str, str | None]:
    """Resolve credentials from the named AWS profile via CLI."""
    try:
        result = subprocess.check_output(
            ["aws", "configure", "export-credentials",
             "--profile", profile, "--format", "env"],
            text=True,
        )
        creds: dict[str, str] = {}
        for line in result.strip().splitlines():
            if line.startswith("export "):
                key, _, val = line[7:].partition("=")
                creds[key.strip()] = val.strip()
        access_key = creds.get("AWS_ACCESS_KEY_ID", "")
        secret_key = creds.get("AWS_SECRET_ACCESS_KEY", "")
        session_token = creds.get("AWS_SESSION_TOKEN")
        if not access_key or not secret_key:
            raise ValueError("Missing credentials in export")
        return access_key, secret_key, session_token
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve AWS credentials for profile '{profile}': {exc}. "
            f"Run: aws sso login --profile {profile}"
        ) from exc


def build_spark(
    warehouse_uri: str,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    shuffle_partitions: int,
) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("IEEEBigData2026-Iceberg-S3-Compaction")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.inMemoryColumnarStorage.enableVectorizedReader", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
        .config("spark.jars.packages", SPARK_PACKAGE)
        # S3A config
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-west-2")
        .config("spark.hadoop.fs.s3a.path.style.access", "false")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        # Iceberg
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", warehouse_uri)
    )
    if session_token:
        builder = builder.config(
            "spark.hadoop.fs.s3a.session.token", session_token
        ).config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
        )
    return builder.getOrCreate()


def cleanup_s3_prefix(bucket: str, prefix: str, profile: str) -> None:
    """Remove any prior experiment data from S3 before starting."""
    try:
        subprocess.run(
            ["aws", "s3", "rm", f"s3://{bucket}/{prefix}/",
             "--recursive", "--profile", profile],
            check=False, capture_output=True,
        )
    except Exception:
        pass


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


def append_batch(spark: SparkSession, batch_index: int, rows_per_batch: int) -> None:
    offset = batch_index * rows_per_batch
    batch = spark.range(rows_per_batch).selectExpr(
        f"format_string('user_%05d', ({offset} + id) % 10000) AS entity_id",
        f"CAST({offset} + id AS BIGINT) AS event_sequence",
        f"CAST((({offset} + id) * 17 + 11) % 1000003 AS BIGINT) AS feature_1",
        f"CAST((({offset} + id) * 31 + 7) % 1000033 AS BIGINT) AS feature_2",
        f"element_at(array('A','B','C','D'), CAST((({offset} + id) % 4) + 1 AS INT)) AS feature_group",
    )
    batch.coalesce(1).writeTo(FULL_TABLE).append()


def table_state(spark: SparkSession) -> dict[str, int]:
    files = spark.sql(
        f"SELECT COUNT(*) AS file_count, COALESCE(SUM(record_count), 0) AS metadata_row_count "
        f"FROM {FULL_TABLE}.files"
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
        f"SELECT COUNT(*) AS row_count, "
        f"SUM(feature_1) AS checksum_1, "
        f"SUM(feature_2) AS checksum_2, "
        f"SUM(LENGTH(entity_id)) AS checksum_entity "
        f"FROM {FULL_TABLE}"
    ).first()
    elapsed = time.perf_counter() - started
    return elapsed, {name: int(row[name]) for name in row.asDict()}


def measure_scans(spark: SparkSession, warmups: int, repetitions: int) -> tuple[list[float], dict[str, int]]:
    checksum = None
    for _ in range(warmups):
        _, checksum = scan_once(spark)
    timings = []
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


def system_metadata(spark: SparkSession, s3_bucket: str) -> dict[str, Any]:
    try:
        memory_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    except Exception:
        memory_bytes = None
    jvm = spark.sparkContext._jvm
    return {
        "backend": "aws-s3",
        "s3_bucket": s3_bucket,
        "s3_region": "us-west-2",
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python_version": platform.python_version(),
        "spark_version": spark.version,
        "iceberg_version": ICEBERG_VERSION,
        "hadoop_aws_version": HADOOP_AWS_VERSION,
        "java_version": jvm.java.lang.System.getProperty("java.version"),
    }


def write_outputs(summary: dict[str, Any], timestamp: str) -> list[Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"iceberg_s3_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "phase", "batch", "file_count", "metadata_row_count",
            "manifest_count", "snapshot_count", "row_count",
            "repetition", "scan_time_s",
        ])
        writer.writeheader()
        writer.writerows(summary["raw_measurements"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoints = summary["checkpoints"]
    pre = [p for p in checkpoints if p["phase"] == "pre_compaction"]
    post = next(p for p in checkpoints if p["phase"] == "post_compaction")

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(
        [p["file_count"] for p in pre],
        [p["timing"]["median_s"] for p in pre],
        marker="o", label="Before compaction (S3)",
    )
    ax.scatter(
        [post["file_count"]], [post["timing"]["median_s"]],
        marker="s", s=55, label="After compaction (S3)",
    )
    ax.set_xlabel("Iceberg data files (count)")
    ax.set_ylabel("Median full-scan latency (seconds)")
    ax.set_title("Amazon S3 scan latency before and after data-file rewrite")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    return [json_path, csv_path, svg_path, pdf_path]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    configure_java_17()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"Resolving AWS credentials (profile: {args.aws_profile})...")
    access_key, secret_key, session_token = resolve_aws_credentials(args.aws_profile)
    print("  OK")

    warehouse_uri = f"s3a://{args.s3_bucket}/{args.s3_prefix}/warehouse"
    print(f"Warehouse: {warehouse_uri}")

    print("Cleaning up prior S3 data...")
    cleanup_s3_prefix(args.s3_bucket, args.s3_prefix, args.aws_profile)

    spark = build_spark(warehouse_uri, access_key, secret_key, session_token, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")

    raw_measurements: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    try:
        create_table(spark)
        environment = system_metadata(spark, args.s3_bucket)

        print(f"Writing {args.batches} commits ({args.rows_per_batch:,} rows each) to S3...")
        last_checksum = None
        for batch_index in range(args.batches):
            append_batch(spark, batch_index, args.rows_per_batch)
            batch_number = batch_index + 1
            if batch_number % args.measure_every == 0 or batch_number == args.batches:
                state = table_state(spark)
                timings, last_checksum = measure_scans(spark, args.warmups, args.repetitions)
                timing_summary = summarize_timings(timings)
                point = {"phase": "pre_compaction", "batch": batch_number, **state, "timing": timing_summary}
                checkpoints.append(point)
                for rep, elapsed in enumerate(timings, start=1):
                    raw_measurements.append({
                        "phase": "pre_compaction", "batch": batch_number, **state,
                        "row_count": last_checksum["row_count"],
                        "repetition": rep, "scan_time_s": elapsed,
                    })
                print(
                    f"  {batch_number:4d} commits | {state['file_count']:4d} files | "
                    f"median scan {timing_summary['median_s']:.3f}s"
                )

        assert last_checksum is not None
        pre_state = table_state(spark)
        pre_timing = checkpoints[-1]["timing"]

        print("Calling Iceberg system.rewrite_data_files on S3...")
        compaction_started = time.perf_counter()
        rewrite_result = spark.sql(
            f"""
            CALL {CATALOG}.system.rewrite_data_files(
                table => '{NAMESPACE}.{TABLE}',
                options => map('rewrite-all', 'true', 'target-file-size-bytes', '134217728')
            )
            """
        ).first()
        compaction_duration = time.perf_counter() - compaction_started

        post_state = table_state(spark)
        post_timings, post_checksum = measure_scans(spark, args.warmups, args.repetitions)
        if post_checksum != last_checksum:
            raise RuntimeError(f"Compaction changed table contents: {last_checksum} != {post_checksum}")
        post_timing = summarize_timings(post_timings)
        checkpoints.append({"phase": "post_compaction", "batch": args.batches, **post_state, "timing": post_timing})
        for rep, elapsed in enumerate(post_timings, start=1):
            raw_measurements.append({
                "phase": "post_compaction", "batch": args.batches, **post_state,
                "row_count": post_checksum["row_count"],
                "repetition": rep, "scan_time_s": elapsed,
            })

        speedup = pre_timing["median_s"] / post_timing["median_s"]
        file_reduction = 100.0 * (pre_state["file_count"] - post_state["file_count"]) / pre_state["file_count"]

        summary = {
            "experiment": "EXP-12: Apache Iceberg small-file compaction on Amazon S3",
            "timestamp_utc": timestamp,
            "scope": (
                "Local Spark + Amazon S3 (us-west-2). Adds genuine per-object S3 GET "
                "request latency absent from local-filesystem and MinIO baselines. "
                "Provides an intermediate data point between local results and cited "
                "production deployments."
            ),
            "parameters": {
                "batches": args.batches,
                "rows_per_batch": args.rows_per_batch,
                "measure_every": args.measure_every,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "shuffle_partitions": args.shuffle_partitions,
                "format_version": 2,
                "s3_bucket": args.s3_bucket,
                "s3_prefix": args.s3_prefix,
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
        print(f"Compaction: {pre_state['file_count']} -> {post_state['file_count']} files ({file_reduction:.1f}%)")
        print(f"Median scan: {pre_timing['median_s']:.3f}s -> {post_timing['median_s']:.3f}s ({speedup:.2f}x)")
        print(f"Compaction duration: {compaction_duration:.3f}s")
        print("Content verification: PASS")
        for p in paths:
            print(f"Saved {p}")
        return summary

    finally:
        spark.stop()


if __name__ == "__main__":
    run_experiment(parse_args())
