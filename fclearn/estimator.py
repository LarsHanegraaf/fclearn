"""Estimators for time series forecasting."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Naive(BaseEstimator, RegressorMixin):
    """Estimator for naive forcasting.

    Data structure of X, should only contain the lags, where the most
    recent lag is the last column.

    Attributes:
        lag (int): Lag that is taken for the Naive forecast
    """

    def __init__(self, lag=1):
        """This estimator can be instantiated with the following parameters.

        Args:
            lag (int): Lag that should be taken for the Naive Forecast, default =1
        """
        self.lag = lag

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
    """

    def __init__(self, window=3):
        """This estimator can be instantiated with the following parameters.

        Args:
            window (int): Amount of lags that the MA should span.
        """
        self.window = window

    def predict(self, X):
        """Predict the next period."""
        return X.iloc[:, -self.window :].mean(axis=1)


class Zero(BaseEstimator, RegressorMixin):
    """Estimator that predicts zero."""

    def predict(self, X):
        """Predict the next period."""
        return np.zeros((X.shape[0], 1))
