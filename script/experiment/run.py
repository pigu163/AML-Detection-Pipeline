"""Run an AML experiment locally or through spark-submit.

Run from the project root, for example:
    uv run python -m script.experiment.run --dataset HI-Small --model xgboost
    uv run python -m script.experiment.run --dataset LI-Small --model catboost \
        --model-params '{"iterations": 300, "depth": 5}' --sample-fraction 0.05

Sampling takes place after feature calculation, so it does not remove history.
The logged validation threshold must be applied to predict_proba for alerts.
"""

import argparse
import json
from time import perf_counter

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from script.data.transaction_features import WINDOWS
from script.experiment.data import load_features
from script.experiment.tracking import (
    add_tracking_arguments,
    configure_tracking,
    log_plan,
    log_runtime,
)
from script.model import catboost, lightgbm, logistic_regression, xgboost
from script.spark import ROOT, add_spark_arguments, create_spark

MODELS = {
    "xgboost": xgboost.create_model,
    "logistic_regression": logistic_regression.create_model,
    "lightgbm": lightgbm.create_model,
    "catboost": catboost.create_model,
}
CATEGORICAL = ["receiving_currency", "payment_currency", "payment_format"]
IDENTIFIERS = ["from_bank", "from_account", "to_bank", "to_account"]


def load_data(
    spark: SparkSession,
    dataset: str,
    sample_fraction: float,
    input_uri: str | None = None,
) -> pd.DataFrame:
    """Read processed data and build features using the full transaction history."""
    started = perf_counter()
    input_uri = input_uri or str(ROOT / "dataset" / "processed" / dataset)
    features = load_features(spark, input_uri).drop(*IDENTIFIERS)
    features = features.where("timestamp is not null and is_laundering is not null")
    # Sampling before feature construction would remove historical transactions.
    if sample_fraction < 1:
        features = features.sample(False, sample_fraction, seed=42)
    log_plan(features)
    # Spark transformations above only build a plan. toPandas is the action that
    # executes reads, windows and joins, then transfers the result to the driver.
    # Measure the combined stage instead of reporting plan-building as compute.
    data = features.toPandas()
    mlflow.log_metrics(
        {
            "feature_collect_seconds": perf_counter() - started,
            "collected_rows": len(data),
        }
    )
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically, keeping transactions with equal timestamps together."""
    data = data.sort_values("timestamp").reset_index(drop=True)
    train_end = data["timestamp"].iloc[int(len(data) * 0.70)]
    validation_end = data["timestamp"].iloc[int(len(data) * 0.85)]
    return (
        data[data["timestamp"] < train_end],
        data[(data["timestamp"] >= train_end) & (data["timestamp"] < validation_end)],
        data[data["timestamp"] >= validation_end],
    )


def build_pipeline(model_name: str, model_params: dict, columns: list[str]) -> Pipeline:
    """Create a classifier and preprocessing fitted only on training data."""
    numeric = [column for column in columns if column not in CATEGORICAL]
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if model_name == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    # User-supplied parameters are passed unchanged to the selected model.
    return Pipeline(
        [("preprocess", preprocessing), ("model", MODELS[model_name](**model_params))]
    )


def choose_threshold(labels: pd.Series, probabilities: np.ndarray) -> float:
    """Select the F1-maximizing threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[np.argmax(scores)])


def run_experiment(data: pd.DataFrame, args: argparse.Namespace) -> str:
    """Train inside the active MLflow run and return the saved model URI."""
    train, validation, test = split_data(data)
    columns = train.columns.drop(["timestamp", "is_laundering"]).tolist()
    model_params = args.model_params.copy()
    # Give each run a reproducible default without overriding the user's seed.
    if not any(key in model_params for key in ("seed", "random_seed", "random_state")):
        model_params["random_seed" if args.model == "catboost" else "random_state"] = 42
    model = build_pipeline(args.model, model_params, columns)

    # Balance class contributions using training labels, not validation or test labels.
    weights = (
        compute_sample_weight(class_weight="balanced", y=train["is_laundering"])
        if args.class_weight == "balanced"
        else None
    )
    started = perf_counter()
    model.fit(train[columns], train["is_laundering"], model__sample_weight=weights)
    mlflow.log_metric("fit_seconds", perf_counter() - started)
    started = perf_counter()
    validation_scores = model.predict_proba(validation[columns])[:, 1]
    threshold = choose_threshold(validation["is_laundering"], validation_scores)
    scores = model.predict_proba(test[columns])[:, 1]
    labels = test["is_laundering"].astype(int)
    predictions = scores >= threshold
    k = min(args.top_k, len(test))
    positives_at_k = labels.iloc[np.argsort(-scores, kind="stable")[:k]].sum()
    metrics = {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "average_precision": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "precision_at_k": positives_at_k / k,
        "recall_at_k": positives_at_k / labels.sum(),
    }
    # Record split boundaries and feature names so the experiment can be reproduced.
    mlflow.log_params(
        {
            "dataset": args.dataset,
            "model": args.model,
            "sample_fraction": args.sample_fraction,
            "sampling_seed": 42,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_end": str(validation["timestamp"].min()),
            "validation_end": str(test["timestamp"].min()),
            "top_k": k,
            "sample_weight": args.class_weight,
        }
    )
    mlflow.log_params({f"model.{key}": value for key, value in model_params.items()})
    mlflow.log_dict(
        {
            "feature_columns": columns,
            "model_parameters": model.named_steps["model"].get_params(),
        },
        "config.json",
    )
    mlflow.log_metrics(
        {
            "validation_threshold": threshold,
            **{f"test_{key}": float(value) for key, value in metrics.items()},
        }
    )
    mlflow.log_dict(
        {
            "labels": [0, 1],
            "matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        },
        "confusion_matrix.json",
    )
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    mlflow.log_figure(fig, "precision_recall.pdf")
    plt.close(fig)
    mlflow.log_metric("evaluation_seconds", perf_counter() - started)
    started = perf_counter()
    example = train[columns].head(5)
    # The model's public interface returns probabilities, exactly as evaluation
    # does. Keep the threshold and input contract with the model, rather than
    # requiring a prediction job to reconstruct them from a separate run metric.
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",
        pyfunc_predict_fn="predict_proba",
        signature=infer_signature(example, model.predict_proba(example)),
        input_example=example,
        metadata={
            "threshold": threshold,
            "feature_columns": columns,
            "positive_class_index": int(np.flatnonzero(model.classes_ == 1)[0]),
            "time_zone": args.time_zone,
            "feature_windows_seconds": WINDOWS,
        },
    )
    mlflow.log_metric("model_logging_seconds", perf_counter() - started)
    mlflow.set_tag("model_uri", model_info.model_uri)
    print(
        f"{args.model}: threshold={threshold:.4f}, "
        + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
        flush=True,
    )
    print(f"Model URI: {model_info.model_uri}", flush=True)
    return model_info.model_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HI-Small", "LI-Small"), required=True)
    parser.add_argument(
        "--input-uri",
        help="Processed Parquet URI; defaults to dataset/processed/<dataset>.",
    )
    parser.add_argument("--model", choices=MODELS, default="xgboost")
    parser.add_argument(
        "--model-params",
        type=json.loads,
        default={},
        help="JSON object of model parameters.",
    )
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="Use 'none' when model parameters already specify class weights.",
    )
    add_spark_arguments(parser)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    args.input_uri = args.input_uri or str(
        ROOT / "dataset" / "processed" / args.dataset
    )

    configure_tracking(args)
    # The run starts before Spark: input/feature failures are recorded as FAILED,
    # and host system metrics include the Spark stage as well as native training.
    with mlflow.start_run(
        run_name=f"{args.dataset}-{args.model}", log_system_metrics=True
    ) as run:
        print(f"Run ID: {run.info.run_id}", flush=True)
        started = perf_counter()
        spark = None
        log_runtime(args)
        try:
            spark = create_spark("AML experiment", args)
            log_runtime(args, spark)
            data = load_data(spark, args.dataset, args.sample_fraction, args.input_uri)
            # Native estimators run on the driver. Stop the Spark session and
            # release its execution resources before fitting on this machine.
            spark.stop()
            spark = None
            run_experiment(data, args)
        finally:
            if spark is not None:
                spark.stop()
            mlflow.log_metric("total_seconds", perf_counter() - started)


if __name__ == "__main__":
    main()
