"""Create a logistic regression classifier."""

from sklearn.linear_model import LogisticRegression


def create_model(**params) -> LogisticRegression:
    """Return an untrained model with optional logistic regression parameters."""
    return LogisticRegression(**params)
