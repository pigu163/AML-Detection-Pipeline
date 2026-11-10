"""Prepare IBM AML transaction CSV files as Parquet."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]
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


def read_transactions(spark: SparkSession, path: Path) -> DataFrame:
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
    destinations = {
        name: DATASET / "processed" / name for name in ("HI-Small", "LI-Small")
    }
    # Spark writes _SUCCESS only after the Parquet output completes.
    pending = [
        name for name, path in destinations.items() if not (path / "_SUCCESS").exists()
    ]
    for name in destinations.keys() - pending:
        print(f"Skipping {name}: Parquet already exists.", flush=True)
    if not pending:
        return

    # Start Spark only when at least one dataset still needs processing.
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("AML data preparation")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for name in pending:
            source = DATASET / name / f"{name}_Trans.csv"
            print(f"Preparing {name}...", flush=True)
            # Overwrite incomplete output; completed datasets were skipped above.
            clean_transactions(read_transactions(spark, source)).write.mode(
                "overwrite"
            ).parquet(str(destinations[name]))
            print(f"Saved {destinations[name]}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
