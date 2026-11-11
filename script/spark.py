"""Shared Spark settings for local jobs and spark-submit applications."""

import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

# Local commands run from the repository root. In a packaged submission, cwd is
# the driver's working directory; a module inside a ZIP has no writable parent.
ROOT = Path.cwd()


def add_spark_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the settings that change between a laptop and a cluster."""
    parser.add_argument(
        "--spark-master",
        help="Optional override, such as local[2] or spark://host:7077.",
    )
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--time-zone", default="Asia/Shanghai")
    parser.add_argument(
        "--event-log-dir", help="Existing local or shared URI for Spark event logs."
    )


def create_spark(name: str, args: argparse.Namespace) -> SparkSession:
    """Use laptop defaults for Python launches and respect spark-submit settings."""
    # A direct Python launch creates the JVM through spark-submit internally.
    # Driver memory must be set here, before the JVM starts. An external
    # spark-submit already owns the gateway and its resource configuration.
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS", "--master local[2] --driver-memory 2g pyspark-shell"
    )
    if "PYSPARK_GATEWAY_PORT" not in os.environ:
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.appName(name)
        .config("spark.sql.session.timeZone", args.time_zone)
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    # Leaving master unset lets an external spark-submit select the cluster.
    if args.spark_master:
        builder = builder.master(args.spark_master)
    if args.event_log_dir:
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", args.event_log_dir
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def parquet_complete(spark: SparkSession, uri: str) -> bool:
    """Check Spark's success marker on the URI's Hadoop-compatible filesystem."""
    # Hadoop selects the filesystem from the URI, so this works with local
    # files and with shared storage such as HDFS. Python Path.exists is local.
    marker = spark._jvm.org.apache.hadoop.fs.Path(f"{uri.rstrip('/')}/_SUCCESS")
    filesystem = marker.getFileSystem(spark._jsc.hadoopConfiguration())
    return filesystem.exists(marker)
