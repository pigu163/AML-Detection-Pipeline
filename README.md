# AML Detection Pipeline

An anti-money laundering engineering project built around PySpark processing, configurable classifiers and MLflow. It demonstrates how transaction history moves through a reproducible batch workflow: preparation, historical computation, training, tracked model artifacts and batch prediction.

![AML detection workflow](image/aml-workflow.svg)

## Approach

The project uses the IBM AML HI-Small and LI-Small datasets, where each record represents a transaction between accounts. PySpark prepares the data and builds transaction-level features from account and counterparty history: transaction frequency, amounts, incoming and outgoing activity, and reciprocal transfers. Historical features use only transactions available before the prediction time.

Chronological training, validation, and test splits preserve the ordering of transactions. A simple baseline provides a reference for XGBoost, while validation data supports model selection and alert-threshold selection. Evaluation focuses on precision, recall, F1, average precision, ROC-AUC, and precision and recall at a given review capacity.

MLflow records the complete experiment, starting before Spark launches. Runs include the effective configuration, code version, stage durations, driver-host resource metrics, evaluation reports and a reusable model artifact. Saved models carry their input signature, probability interface and alert threshold. Batch prediction reuses the historical computation and scores transactions in Spark partitions.

## Engineering design

| Component | Responsibility |
| --- | --- |
| PySpark | CSV-to-Parquet preparation, historical windows, joins and partitioned batch prediction |
| Model factories | Independent XGBoost, Logistic Regression, LightGBM and CatBoost constructors with caller-supplied parameters |
| MLflow | Run lifecycle, configuration and environment provenance, system metrics, plans, reports and model artifacts |
| Batch prediction | Load a tracked model, apply its saved input/decision contract and write scored transactions |

The execution layer shares one Spark factory. Direct Python launches use `local[2]` for a MacBook Air; `spark-submit` can select a configured Linux Spark cluster. DataFrame processing and batch prediction can execute on workers. Native classifier training runs on the driver and requires an in-memory training matrix. A single Linux server can also use local parallelism through `local[N]`.

Spark AQE and Arrow conversion are enabled. Physical plans and optional event logs expose execution details; MLflow records stage timings and host resource usage. Local metadata uses SQLite with an absolute artifact location. Shared deployments can use an MLflow tracking server and shared Spark filesystem URIs.

## Run locally

```bash
uv sync --locked
uv run --locked python -m script.data.prepare_transactions
uv run --locked python -m script.experiment.run \
  --dataset HI-Small --model xgboost --sample-fraction 0.05 \
  --model-params '{"n_estimators": 200, "max_depth": 6, "n_jobs": 2}'
```

The command prints the MLflow run ID and saved model URI. See [the running guide](docs/running.md) for prerequisites, resource settings, the MLflow UI, batch prediction, output overwrite behavior and Linux cluster submission.

## Interpretation

The model scores transactions for review; a score or behavioral pattern is not proof of money laundering. The datasets are synthetic, and their results should be interpreted within that setting.

## Tools

Python · PySpark · XGBoost · Logistic Regression · LightGBM · CatBoost · MLflow
