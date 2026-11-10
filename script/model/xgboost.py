"""Create an XGBoost classifier."""

from xgboost import XGBClassifier


def create_model(**params) -> XGBClassifier:
    """Return an untrained model with optional XGBoost parameters."""
    return XGBClassifier(**params)
