# AGENTS.md — AML Detection with PySpark, XGBoost and MLflow

## 1. Project Objective

Build a reproducible end-to-end Anti-Money Laundering (AML) detection pipeline using the IBM AML synthetic transaction datasets (HI-Small and LI-Small).

The project should demonstrate practical proficiency in:

- **PySpark** for transaction data processing and feature engineering.
- **XGBoost** for supervised AML detection.
- **MLflow** for experiment tracking, model evaluation, and model management.
- **SHAP** for model interpretability.
- **pytest, Ruff, and GitHub Actions** for code quality and automated testing.

Keep the project lightweight, reproducible, and suitable for local development. Avoid unnecessary infrastructure and excessive architectural complexity.

## 2. Core Tasks

### Task 1 — Data Exploration and Processing

Use PySpark to load and process the IBM AML datasets.

- Inspect schemas, feature distributions, missing values, and duplicate records.
- Analyze transaction amounts, timestamps, currencies, payment formats, and account activities.
- Examine class imbalance and differences between HI-Small and LI-Small.
- Apply appropriate data cleaning and type conversions.
- Use Spark DataFrame APIs and Spark SQL where appropriate.
- Store processed datasets in Parquet format.

### Task 2 — Transaction Feature Engineering

Implement behavioral and temporal features using PySpark.

Focus on:

- Transaction frequency and velocity over historical time windows.
- Historical transaction amount statistics.
- Incoming and outgoing transaction counts and amounts.
- Unique counterparties and counterparty diversity.
- Fan-in and fan-out transaction patterns.
- Reciprocal transaction behavior.
- Time intervals between transactions.
- Relevant currency, payment format, and temporal features.

Use PySpark aggregations, joins, and window functions.

**Critical requirement:** Avoid temporal data leakage. Features for each transaction must only depend on information available at its prediction time.

### Task 3 — XGBoost AML Detection

Develop a supervised binary classification pipeline using XGBoost.

- Establish a simple baseline model for comparison.
- Use chronological train, validation, and test splits.
- Handle class imbalance appropriately.
- Train XGBoost using engineered transaction features.
- Perform lightweight hyperparameter optimization.
- Optimize classification thresholds on validation data.
- Evaluate the final model on held-out test data.

Prioritize reproducibility, appropriate validation, and meaningful model comparisons rather than maximizing accuracy.

### Task 4 — Model Evaluation

Evaluate performance using metrics relevant to AML detection.

Required metrics:

- Precision, Recall, and F1-score.
- PR-AUC (Average Precision).
- ROC-AUC.
- Precision@K and Recall@K.
- Confusion Matrix.

Include:

- Precision-Recall curves.
- Comparison against baseline models.
- Analysis of false positives and false negatives.
- Investigation of model performance under different alert-review capacities.

Accuracy alone must not be used as the primary evaluation metric.

### Task 5 — MLflow Experiment Tracking

Integrate MLflow into the training and evaluation workflow.

Track:

- Dataset identifiers and experiment configurations.
- Feature configurations and data split information.
- XGBoost hyperparameters.
- Training and evaluation metrics.
- Model artifacts.
- Evaluation figures and reports.

Enable comparison between different model runs and experiments.

The complete training workflow should be reproducible from a documented configuration.

### Task 6 — Model Interpretability

Use SHAP to explain XGBoost predictions.

Implement:

- Global feature importance.
- SHAP summary plots.
- Individual transaction explanations.
- Analysis of influential transaction behavior features.

Discuss whether important features are consistent with plausible AML detection patterns.

Do not interpret model predictions as definitive evidence of money laundering.

### Task 7 — Cross-Dataset Experiments

Use HI-Small and LI-Small to investigate model robustness.

Conduct:

1. Train and evaluate on HI-Small using chronological splits.
2. Train and evaluate on LI-Small using chronological splits.
3. Train on HI-Small and evaluate on LI-Small, where feature compatibility and data separation permit.

Compare model performance, class imbalance, feature distributions, and generalization behavior.

Avoid unsupported claims about distribution shift or domain adaptation.

Track all experiments using MLflow.

### Task 8 — Testing and Automation

Maintain basic software engineering standards.

- Use pytest for feature engineering and preprocessing tests.
- Validate temporal feature correctness and absence of future-data leakage.
- Use Ruff for Python linting and formatting.
- Configure GitHub Actions to execute automated tests and code checks.
- Use fixed random seeds where applicable.
- Ensure local execution is reproducible.

CI should rely on small synthetic test fixtures rather than requiring the complete IBM datasets.

## 3. Technical Constraints

- Use Python as the primary programming language.
- Use PySpark for data processing and feature engineering.
- Use XGBoost for model training.
- Use MLflow for experiment tracking.
- Prefer native XGBoost training after Spark feature processing for datasets that fit in memory.
- Use Spark Local Mode when appropriate.
- Optimize resource usage for a local machine with approximately 8 GB RAM.
- Avoid unnecessary services such as Kafka, Flink, Kubernetes, and Neo4j.
- Docker is optional and should only be used when it simplifies reproducibility.
- Do not introduce GNNs or LLM-based components.

## 4. Implementation Principles

- Prioritize simplicity, readability, and reproducibility.
- Prefer established libraries over custom implementations.
- Use modular, testable functions.
- Avoid excessive abstractions and unnecessary dependencies.
- Ensure the same preprocessing and feature transformations are applied during training and evaluation.
- Clearly distinguish transaction-level predictions from account-level risk assessments.
- Prevent data leakage across training and evaluation stages.
- Do not fabricate results, benchmarks, or performance claims.
- Use the actual IBM AML dataset schema rather than assuming field names or labels.

## 5. Expected Outcomes

The completed project should demonstrate:

1. Practical PySpark experience with financial transaction data.
2. Behavioral and temporal feature engineering for AML detection.
3. XGBoost modeling on highly imbalanced data.
4. Appropriate model evaluation based on AML alert prioritization.
5. Experiment tracking and model management using MLflow.
6. Model interpretability using SHAP.
7. Reproducible workflows with automated code quality checks.

The GitHub README should clearly communicate the problem, methodology, experiments, evaluation results, and limitations.

## 6. Priority

Implement in this order:

**Phase 1 — Core Pipeline**
- Data processing with PySpark.
- Behavioral feature engineering.
- XGBoost training and evaluation.
- MLflow experiment tracking.

**Phase 2 — Model Analysis**
- Hyperparameter optimization.
- Threshold optimization.
- SHAP interpretability.
- AML-specific evaluation.

**Phase 3 — Robustness and Engineering**
- HI-Small and LI-Small comparison.
- Cross-dataset evaluation.
- pytest and Ruff.
- GitHub Actions.
- Final documentation and experiment results.

Complete the core pipeline before introducing optional enhancements.