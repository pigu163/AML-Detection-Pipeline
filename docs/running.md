# Running the pipeline

## Execution model

The same Spark DataFrame code runs in local mode on a Mac and can be submitted to a configured Spark cluster on Linux. Native XGBoost, Logistic Regression, LightGBM and CatBoost training runs on the **driver**. Batch prediction uses `mapInPandas` to score Arrow batches on Spark workers.

| Environment | Spark execution | Native training |
| --- | --- | --- |
| MacBook Air | `local[2]`, 2 GiB driver JVM heap by default | Local Python process |
| One Linux server | `local[N]` uses that server's cores | Driver Python process |
| Spark cluster | Workers execute DataFrame processing and batch prediction | Driver Python process |

A large single server provides local parallelism. Execution across machines requires a Spark cluster. The cluster commands below are submission examples; validation for this repository is performed locally.

## Environment

Run commands from the repository root. Install Python 3.12 and a Java version supported by the locked Spark release, then create the environment:

```bash
uv sync --locked
```

The lock file includes `psutil`, which MLflow uses for CPU, memory, disk and network metrics. A directly launched local job selects the current Python interpreter for its Python workers. With `spark-submit`, configure the Python environment on all workers explicitly.

## Prepare Parquet

```bash
uv run --locked python -m script.data.prepare_transactions
```

Input is read from `dataset/HI-Small/HI-Small_Trans.csv` and the corresponding LI-Small directory. Output goes to `dataset/processed/<dataset>/`. A completed `_SUCCESS` marker skips an existing output. To rebuild after changing preprocessing:

```bash
uv run --locked python -m script.data.prepare_transactions \
  --dataset HI-Small --overwrite
```

`--input-root` and `--output-root` also accept shared Spark filesystem URIs. Both contain dataset subdirectories. CSV columns are mapped by position to distinguish the duplicate `Account` headers.

## Train and track

```bash
uv run --locked python -m script.experiment.run \
  --dataset HI-Small --model xgboost \
  --model-params '{"n_estimators": 200, "max_depth": 6, "n_jobs": 2}' \
  --sample-fraction 0.05
```

Other model names are `logistic_regression`, `lightgbm` and `catboost`. `--input-uri` overrides the processed Parquet location. Sampling occurs after historical feature calculation. The default fraction is `1.0`; choose a smaller fraction when the driver cannot hold the collected training matrix. Numeric model inputs are converted to float32 in both training and prediction; the source Parquet retains its original precision.

MLflow starts before Spark, so input and feature failures belong to the same run. The command prints a run ID and, on success, a model URI such as `models:/m-...`. Each run records:

- `run_config.json`: CLI options, Git commit, dirty-worktree status when available, library versions, Java version, effective Spark configuration and feature windows.
- `feature_plan.txt`: the formatted physical plan, recorded without an extra Spark action.
- Stage durations, collected row count, evaluation metrics, PDF PR curve and confusion matrix.
- Driver-host system metrics sampled every five seconds. Spark worker and shuffle metrics are inspected through Spark UI or event logs.
- The fitted preprocessing/classifier, input signature and example, probability output interface, and threshold metadata.

`feature_collect_seconds` includes reading, Spark computation and transfer to Pandas. Spark is lazy: timing the construction of a DataFrame alone would not measure this work. The Spark session is stopped before native training to release its execution resources.

Balanced training sample weights are enabled by default. If the model parameters already configure class weights, use `--class-weight none`. A prediction consumer applies the saved threshold to the saved positive-class probability column.

## MLflow storage and UI

Local runs use an absolute SQLite URI pointing to `mlruns/mlflow.db`. New local experiments use an absolute artifact location under `mlruns/artifacts/`; metadata and artifact paths do not depend on a later UI process's working directory.

```bash
uv run --locked mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlruns/mlflow.db" \
  --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. An existing experiment keeps the artifact location it was created with. To choose another location, create a new `--experiment-name` with `--artifact-root`.

For a shared tracking server, pass `--tracking-uri http://mlflow-host:5000` or set `MLFLOW_TRACKING_URI`. Let that server manage its artifact location. Drivers and scoring workers must be able to access it and its artifacts. SQLite remains suitable for this local demonstration; a separate database service is not required.

## Batch prediction

Use a model produced by the training entry point above:

```bash
uv run --locked python -m script.experiment.predict \
  --model-uri models:/REPLACE_WITH_MODEL_ID \
  --input-uri dataset/processed/HI-Small \
  --output-uri mlruns/predictions/HI-Small \
  --from-time '2022-09-15 00:00:00'
```

The input includes available transaction history. `--from-time` is inclusive and `--to-time` is exclusive; target filters are applied after historical features are defined. Prediction inherits the time zone stored in the model.

Each Spark task loads the model once and scores Arrow batches. Output contains transaction time, sender/recipient identifiers, `risk_score` and `is_alert`. The output URI is **overwritten** by the job. Features are not saved. An `Observation` counts rows and alerts during the write, so a second action does not repeat inference. The prediction run records its model URI, threshold, output location, execution plan, configuration and duration.

## Spark settings and execution evidence

Shared options for preparation, training and prediction:

| Option | Default | Purpose |
| --- | --- | --- |
| `--spark-master` | Unset | Preserve the master chosen by `spark-submit`; direct Python launches default to `local[2]` |
| `--shuffle-partitions` | `8` | Laptop starting point; adjust for cluster size and measured task sizes |
| `--time-zone` | `Asia/Shanghai` | Match the convention used by the existing Parquet; prediction uses the model's saved value |
| `--event-log-dir` | Unset | Existing URI for Spark event logs |

AQE and Arrow conversion are enabled by the shared factory. Partition count is a starting configuration, not a performance guarantee. Compare stage duration, shuffle size and spill using Spark UI at the URL recorded in `run_config.json`. To retain execution history after the application exits:

```bash
mkdir -p mlruns/spark-events
uv run --locked python -m script.experiment.run \
  --dataset HI-Small --sample-fraction 0.05 \
  --event-log-dir "file://$PWD/mlruns/spark-events"
```

A Spark History Server can read that event-log directory. Configure a shared event-log URI when using a cluster. The code does not cache entire feature tables automatically; use the recorded plan and Spark UI to identify repeated work before introducing caching.

Driver heap must be configured before the JVM starts. For a direct Python launch, override the local defaults with `PYSPARK_SUBMIT_ARGS`:

```bash
PYSPARK_SUBMIT_ARGS='--master local[2] --driver-memory 3g pyspark-shell' \
  uv run --locked python -m script.experiment.run --dataset HI-Small --sample-fraction 0.05
```

## Linux and cluster submission

On one Linux server, `--spark-master 'local[8]'` uses eight local task slots. For a multi-machine Spark cluster, distribute the Python source package and use shared input/output URIs. From the repository root, create a source archive containing only Python files:

```bash
uv run --locked python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

with ZipFile('/tmp/aml-code.zip', 'w') as archive:
    for path in Path('script').rglob('*.py'):
        archive.write(path, path.as_posix())
PY
```

Example for a configured Spark standalone cluster, using client deployment:

```bash
PYSPARK_PYTHON=/opt/aml/.venv/bin/python \
uv run --locked spark-submit \
  --master spark://spark-master:7077 --deploy-mode client \
  --driver-memory 4g --executor-memory 4g --executor-cores 2 \
  --py-files /tmp/aml-code.zip \
  script/experiment/run.py \
  --dataset HI-Small --input-uri hdfs://namenode:9000/aml/processed/HI-Small \
  --shuffle-partitions 200 --sample-fraction 0.05 \
  --tracking-uri http://mlflow-host:5000 \
  --code-version "$(git rev-parse HEAD)"
```

Replace hosts, paths and resource settings with those of your environment. The driver and workers need compatible Java/Spark/Python versions and the locked Python dependencies; the example assumes the same Python environment path on each host. `--py-files` distributes project source, not third-party dependencies. In packaged submissions, `--code-version` supplies provenance when the driver has no Git checkout.

For distributed prediction, submit `script/experiment/predict.py` with the same Spark and packaging options, a model URI from the shared MLflow server, and shared `--input-uri`/`--output-uri`. A local Mac file path or SQLite database is not shared storage for remote workers.

The native training stage still requires a matrix that fits the driver. Increasing Spark workers accelerates eligible processing and scoring stages; it does not make these four native estimators distributed trainers.
