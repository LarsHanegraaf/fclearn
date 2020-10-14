"""Estimators for time series forecasting."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Naive(BaseEstimator, RegressorMixin):
    """Estimator for naive forcasting.

    Data structure of X, should only contain the lags, where the most
    recent lag is the last column.

    Attributes:
        lag (int): Lag that is taken for the Naive forecast

    Args:
        lag (int): Lag that should be taken for the Naive Forecast, default =1
    """

    def __init__(self, lag=1):
        """Constructor."""
        self.lag = lag

    def fit(self, X, y):
        """Fit the estimator on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def predict(self, X):
        """Predict the next period."""
        return X.iloc[:, -self.lag]


class MovingAverage(BaseEstimator, RegressorMixin):
    """Estimator for a Moving Average.

    Moving Average estimator which returns the average of lag t-1 up untill t-*window '
    size*.
    The lags should be columns that are created during preprocessing and
    should have column names in the form of 't-*lag number*' (e.g. 't-1').

    Attributes:
        window (int): Amount of lags that the MA spans.

    Args:
            window (int): Amount of lags that the MA should span.
    """

    def __init__(self, window=3):
        """Constructor."""
        self.window = window

    def fit(self, X, y):
        """Fit the estimator on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def predict(self, X):
        """Predict the next period."""
        return X.iloc[:, -self.window :].mean(axis=1)


class Zero(BaseEstimator, RegressorMixin):
    """Estimator that predicts zero."""

    def fit(self, X, y):
        """Fit the estimator on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def predict(self, X):
        """Predict the next period."""
        return np.zeros((X.shape[0], 1))
