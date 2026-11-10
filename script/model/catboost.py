"""Create a CatBoost classifier."""

from catboost import CatBoostClassifier


def create_model(**params) -> CatBoostClassifier:
    """Return an untrained model with optional CatBoost parameters."""
    return CatBoostClassifier(**params)
