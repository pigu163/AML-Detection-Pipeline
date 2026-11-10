# AML-Detection-Pipeline
An end-to-end Anti-Money Laundering (AML) detection pipeline using PySpark for transaction feature engineering, XGBoost for risk prediction, and MLflow for experiment tracking, with SHAP-based explainability.

## Prepare transaction data

With Java and the project dependencies installed, run:

```bash
uv run python script/data/prepare_transactions.py
```

The script reads the HI-Small and LI-Small transaction CSV files from `dataset/` and writes Parquet datasets to `dataset/processed/HI-Small/` and `dataset/processed/LI-Small/`. Completed outputs are skipped on subsequent runs. Remove the relevant processed directory to regenerate it after changing the source or preprocessing.

Bank and account identifiers remain strings. Missing currency and payment format values become `Unknown`; missing account identifiers, timestamps, amounts, and labels remain null.

## Calculate transaction features

```python
from pyspark.sql import SparkSession
from script.data.transaction_features import transaction_features

spark = SparkSession.builder.master("local[2]").appName("AML features").getOrCreate()
transactions = spark.read.parquet("dataset/processed/HI-Small")
features = transaction_features(transactions)
```

The function returns a lazy Spark DataFrame and does not save it. Run HI-Small and LI-Small separately, using the complete history for each dataset. Account and pair features use historical windows ending before the current transaction time; distinct counterparties use a one-day window. Spark's `day_of_week` is 1 for Sunday through 7 for Saturday. The input `is_laundering` column remains available as the label and must be excluded from model features.
