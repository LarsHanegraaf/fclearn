"""Estimators for time series forecasting."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Naive(BaseEstimator, RegressorMixin):
    """Estimator for naive forcasting.

    Data structure of X, should only contain the lags, where the most
    recent lag is the last column.

    Attributes:
        lag (type):

    Inheritance:
        BaseEstimator:
        RegressorMixin:

    Args:
        lag=1 (undefined):

    """

    def __init__(self, lag=1):
        """Constructor."""
        self.lag = lag

    def fit(self, X, y=None):
        """Standard sklearn fit function.""",
        return self

    def predict(self, X):
        """Standard sklearn predict fuction."""
        return X.iloc[:, -self.lag]


class MovingAverage(BaseEstimator, RegressorMixin):
    """Estimator for a Moving Average.

    Moving Average estimator which returns the average of lag t-1 up untill t-*window size*.
    The lags should be columns that are created during preprocessing and
    should have column names in the form of 't-*lag number*' (e.g. 't-1').

    Attributes:
        window (type):

    Inheritance:
        BaseEstimator:
        RegressorMixin:

    Args:
        window=3 (undefined):

    """

    def __init__(self, window=3):
        """Constructor."""
        self.window = window

    def fit(self, X, y=None):
        """Standard sklearn fit function.""",
        return self

    def predict(self, X):
        """Standard sklearn predict fuction."""
        return X.iloc[:, -self.window :].mean(axis=1)


class Zero(BaseEstimator, RegressorMixin):
    """Description of Zero.

    Attributes:
        window (type):

    Inheritance:
        BaseEstimator:
        RegressorMixin:

    Args:
        window=3 (undefined):

    """

    def __init__(self, window=3):
        """Constructor."""
        self.window = window

    def fit(self, X, y=None):
        """Standard sklearn fit function.""",
        return self

    def predict(self, X):
        """Standard sklearn predict fuction."""
        return np.zeros((X.shape[0], 1))
