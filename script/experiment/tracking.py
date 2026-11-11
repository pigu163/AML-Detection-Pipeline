"""Small MLflow helpers shared by training and batch prediction."""

import argparse
import io
import os
import platform
import subprocess
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

import mlflow

from script.data.transaction_features import WINDOWS
from script.spark import ROOT


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-uri",
        help="MLflow server URI or SQLite URI; also accepts MLFLOW_TRACKING_URI.",
    )
    parser.add_argument("--experiment-name", default="aml-detection")
    parser.add_argument(
        "--artifact-root",
        help="Artifact directory or URI, used when creating an experiment.",
    )
    parser.add_argument(
        "--code-version",
        default=os.environ.get("GIT_COMMIT"),
        help="Commit SHA for packaged cluster submissions.",
    )


def configure_tracking(args: argparse.Namespace) -> None:
    """Keep metadata in SQLite and artifacts at an explicit absolute location."""
    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        directory = ROOT / "mlruns"
        directory.mkdir(exist_ok=True)
        uri = f"sqlite:///{directory / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    artifact_root = args.artifact_root
    if artifact_root is None and uri.startswith("sqlite:"):
        artifact_root = (ROOT / "mlruns" / "artifacts").as_uri()
    if artifact_root and "://" not in artifact_root:
        artifact_root = Path(artifact_root).resolve().as_uri()
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(args.experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(args.experiment_name)
    # A short sampling interval makes CPU and memory visible in local demos.
    # These are driver-host metrics; executor/shuffle details live in Spark UI.
    mlflow.set_system_metrics_sampling_interval(5)
    mlflow.set_system_metrics_samples_before_logging(1)


def log_runtime(args: argparse.Namespace, spark=None) -> None:
    """Save requested options first, then enrich them with effective Spark settings."""
    code_version = args.code_version
    if code_version is None and (Path.cwd() / ".git").exists():
        code_version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    config = {
        "arguments": vars(args),
        "code_version": code_version or "unknown",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "pyspark",
                "mlflow",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "catboost",
                "psutil",
            )
        },
        "feature_windows_seconds": WINDOWS,
    }
    if (Path.cwd() / ".git").exists():
        config["working_tree_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    if spark is not None:
        keys = (
            "spark.sql.session.timeZone",
            "spark.sql.shuffle.partitions",
            "spark.sql.adaptive.enabled",
            "spark.sql.execution.arrow.pyspark.enabled",
        )
        config["spark"] = {key: spark.conf.get(key) for key in keys}
        config["spark"].update(
            {
                "master": spark.sparkContext.master,
                "application_id": spark.sparkContext.applicationId,
                "ui_url": spark.sparkContext.uiWebUrl,
                "java_version": spark.sparkContext._jvm.java.lang.System.getProperty(
                    "java.version"
                ),
                "driver_heap_bytes": spark.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory(),
                "launch_configuration": {
                    key: value
                    for key, value in spark.sparkContext.getConf().getAll()
                    if key
                    in (
                        "spark.driver.memory",
                        "spark.executor.memory",
                        "spark.executor.cores",
                        "spark.executor.instances",
                        "spark.eventLog.enabled",
                        "spark.eventLog.dir",
                    )
                },
            }
        )
        mlflow.set_tag("spark.master", spark.sparkContext.master)
    mlflow.set_tag("code.version", config["code_version"])
    mlflow.log_dict(config, "run_config.json")


def log_plan(frame, name: str = "feature_plan.txt") -> None:
    """Capture Spark's physical plan without adding a separate Spark action."""
    output = io.StringIO()
    with redirect_stdout(output):
        frame.explain(mode="formatted")
    mlflow.log_text(output.getvalue(), name)
