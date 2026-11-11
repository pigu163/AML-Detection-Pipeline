"""Build the same Spark inputs for training and batch prediction."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from script.data.transaction_features import transaction_features


def load_features(spark: SparkSession, input_uri: str) -> DataFrame:
    """Read processed transactions and return a lazy feature DataFrame."""
    transactions = spark.read.parquet(input_uri)
    features = transaction_features(transactions)
    # Native estimators consume floating-point matrices. Use float32 consistently
    # in both entry points to reduce driver/worker memory; source Parquet is kept
    # in its original precision. The label is kept as an integer, not a feature.
    return features.select(
        *[
            F.col(field.name).cast("float").alias(field.name)
            if isinstance(field.dataType, NumericType) and field.name != "is_laundering"
            else F.col(field.name)
            for field in features.schema.fields
        ]
    )
