"""
Iris dataset loader.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load and split the Iris dataset.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, y_train, X_test, y_test
