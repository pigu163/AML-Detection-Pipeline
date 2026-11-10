"""Create a LightGBM classifier."""

from lightgbm import LGBMClassifier


def create_model(**params) -> LGBMClassifier:
    """Return an untrained model with optional LightGBM parameters."""
    return LGBMClassifier(**params)
