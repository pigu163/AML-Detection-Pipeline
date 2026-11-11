"""Prepare IBM AML transaction CSV files as Parquet."""

import argparse
import sys
from pathlib import Path

# Keep the original direct-script command working alongside python -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from script.spark import ROOT, add_spark_arguments, create_spark, parquet_complete

DATASET = ROOT / "dataset"
# The CSV has two columns named Account, so the order must match the source file.
COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def read_transactions(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read the duplicate Account headers by their column positions."""
    return spark.read.option("header", True).csv(str(path)).toDF(*COLUMNS)


def clean_transactions(transactions: DataFrame) -> DataFrame:
    """Convert transaction types and fill missing categories."""
    # Keep account and bank IDs as strings to preserve leading zeroes.
    # Leave missing IDs, amounts, timestamps and labels null.
    return (
        transactions.withColumn(
            "timestamp", F.to_timestamp("timestamp", "yyyy/MM/dd HH:mm")
        )
        .withColumn("amount_received", F.col("amount_received").cast("double"))
        .withColumn("amount_paid", F.col("amount_paid").cast("double"))
        .withColumn("is_laundering", F.col("is_laundering").cast("int"))
        .fillna(
            "Unknown",
            subset=["receiving_currency", "payment_currency", "payment_format"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("HI-Small", "LI-Small", "all"), default="all"
    )
    parser.add_argument(
        "--input-root",
        default=str(DATASET),
        help="Root containing each dataset's CSV directory; accepts shared URIs.",
    )
    parser.add_argument("--output-root", default=str(DATASET / "processed"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild completed Parquet output after changing preprocessing.",
    )
    add_spark_arguments(parser)
    args = parser.parse_args()
    spark = create_spark("AML data preparation", args)
    try:
        names = ("HI-Small", "LI-Small") if args.dataset == "all" else (args.dataset,)
        for name in names:
            destination = f"{args.output_root.rstrip('/')}/{name}"
            # Check the success marker on Spark's filesystem, which may be remote.
            if not args.overwrite and parquet_complete(spark, destination):
                print(f"Skipping {name}: Parquet already exists.", flush=True)
                continue
            source = f"{args.input_root.rstrip('/')}/{name}/{name}_Trans.csv"
            print(f"Preparing {name}...", flush=True)
            # Completed output is retained unless --overwrite was requested.
            clean_transactions(read_transactions(spark, source)).write.mode(
                "overwrite"
            ).parquet(destination)
            print(f"Saved {destination}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
