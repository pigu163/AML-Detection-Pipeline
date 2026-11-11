"""Build historical features and score transactions in Spark partitions.

Run from the project root:
    uv run python -m script.experiment.predict --model-uri models:/<model-id> \
        --input-uri dataset/processed/HI-Small --output-uri mlruns/predictions

Input must include the available transaction history. Optional target time
filters are applied after feature construction, so earlier history is retained.
"""

import argparse
from time import perf_counter

import mlflow
import mlflow.sklearn
import numpy as np
from pyspark.sql import Observation
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StructField, StructType

from script.experiment.data import load_features
from script.experiment.tracking import (
    add_tracking_arguments,
    configure_tracking,
    log_plan,
    log_runtime,
)
from script.spark import add_spark_arguments, create_spark

LOCATORS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account"]


def score_transactions(spark, args: argparse.Namespace, metadata: dict):
    """Return a lazy, partitioned scoring job without collecting it to the driver."""
    columns = metadata["feature_columns"]
    threshold = metadata["threshold"]
    positive_index = metadata["positive_class_index"]
    frame = load_features(spark, args.input_uri)
    # Select target transactions only after the full history has been supplied.
    if args.from_time:
        frame = frame.where(F.col("timestamp") >= F.to_timestamp(F.lit(args.from_time)))
    if args.to_time:
        frame = frame.where(F.col("timestamp") < F.to_timestamp(F.lit(args.to_time)))
    frame = frame.select(*LOCATORS, *columns)
    log_plan(frame, "prediction_plan.txt")
    output_schema = StructType(
        [frame.schema[name] for name in LOCATORS]
        + [
            StructField("risk_score", DoubleType()),
            StructField("is_alert", BooleanType()),
        ]
    )
    model_uri = args.model_uri
    tracking_uri = mlflow.get_tracking_uri()

    def score_batches(batches):
        # Each executor task loads the saved preprocessing/model once, then
        # processes Arrow batches. The whole dataset never becomes one Pandas
        # DataFrame. Cluster workers need the same Python dependencies and access
        # to the MLflow server and artifact storage used by this model URI.
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.sklearn.load_model(model_uri)
        for batch in batches:
            scores = model.predict_proba(batch[columns])[:, positive_index]
            result = batch[LOCATORS].copy()
            result["risk_score"] = np.asarray(scores, dtype="float64")
            result["is_alert"] = scores >= threshold
            yield result

    return frame.mapInPandas(score_batches, schema=output_schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument(
        "--input-uri",
        required=True,
        help="Processed Parquet containing target transactions and their history.",
    )
    parser.add_argument(
        "--output-uri",
        required=True,
        help="Parquet destination; overwritten by this job.",
    )
    parser.add_argument(
        "--from-time", help="Inclusive target timestamp in the model's saved time zone."
    )
    parser.add_argument(
        "--to-time", help="Exclusive target timestamp in the model's saved time zone."
    )
    add_spark_arguments(parser)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    configure_tracking(args)
    with mlflow.start_run(run_name="batch-predict", log_system_metrics=True) as run:
        print(f"Run ID: {run.info.run_id}", flush=True)
        started = perf_counter()
        spark = None
        log_runtime(args)
        try:
            # Inference inherits the training convention so hour/day features
            # have identical meanings on a Mac or on a Linux executor.
            metadata = mlflow.models.get_model_info(args.model_uri).metadata
            args.time_zone = metadata["time_zone"]
            spark = create_spark("AML batch prediction", args)
            log_runtime(args, spark)
            predictions = score_transactions(spark, args, metadata)
            observation = Observation("predictions")
            # Observation counts rows during the write itself. A separate count
            # action would recompute both features and model predictions.
            predictions.observe(
                observation,
                F.count(F.lit(1)).alias("rows"),
                F.coalesce(F.sum(F.col("is_alert").cast("long")), F.lit(0)).alias(
                    "alerts"
                ),
            ).write.mode("overwrite").parquet(args.output_uri)
            mlflow.log_metrics(
                {name: float(value) for name, value in observation.get.items()}
            )
            mlflow.log_params(
                {
                    "model_uri": args.model_uri,
                    "output_uri": args.output_uri,
                    "threshold": metadata["threshold"],
                }
            )
            print(f"Predictions saved to {args.output_uri}", flush=True)
        finally:
            if spark is not None:
                spark.stop()
            mlflow.log_metric("total_seconds", perf_counter() - started)


if __name__ == "__main__":
    main()
