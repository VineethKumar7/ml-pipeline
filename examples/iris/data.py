"""
Iris Dataset Loader
===================

This module provides the data loading function for the Iris classifier example.
It demonstrates the expected interface for data loaders in the ML Pipeline.

The Iris dataset is a classic machine learning dataset containing:
- 150 samples of iris flowers
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: setosa, versicolor, virginica

Data Loading Contract:
    All data loader modules should provide a load_data() function that returns:
    (X_train, y_train, X_test, y_test)

    Where:
    - X_train: Training features (n_train_samples, n_features)
    - y_train: Training labels (n_train_samples,)
    - X_test: Test features (n_test_samples, n_features)
    - y_test: Test labels (n_test_samples,)

Usage:
    from examples.iris.data import load_data

    X_train, y_train, X_test, y_test = load_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

See Also:
    - examples/iris/model.py: Model implementation using this data
    - src/training/train.py: How the training script uses load_data()
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load and split the Iris dataset for training.

    Loads the classic Iris flower dataset from scikit-learn and
    splits it into training and test sets using stratified sampling
    to maintain class proportions.

    Args:
        test_size: Fraction of data to use for testing (default: 0.2)
                  Must be between 0 and 1.

        random_state: Random seed for reproducible splits (default: 42)
                     Use the same seed to get identical splits.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test):

        - X_train: Training features, shape (n_train, 4)
                  Each row contains [sepal_length, sepal_width,
                  petal_length, petal_width]

        - y_train: Training labels, shape (n_train,)
                  Integer class labels: 0=setosa, 1=versicolor, 2=virginica

        - X_test: Test features, shape (n_test, 4)

        - y_test: Test labels, shape (n_test,)

    Example:
        # Default split (80% train, 20% test)
        X_train, y_train, X_test, y_test = load_data()

        # Custom split (70% train, 30% test)
        X_train, y_train, X_test, y_test = load_data(test_size=0.3)

        # Check shapes
        print(f"X_train shape: {X_train.shape}")  # (120, 4)
        print(f"y_train shape: {y_train.shape}")  # (120,)

    Note:
        The stratify parameter ensures each class is proportionally
        represented in both train and test sets. This is important
        for imbalanced datasets, though Iris is balanced (50 per class).
    """
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Maintain class proportions in split
    )

    return X_train, y_train, X_test, y_test
