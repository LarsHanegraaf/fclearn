import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Naive(BaseEstimator, RegressorMixin):
    """
    Naive estimator, which uses the value of the specified lag as the prediciton.
    The lags should be columns that are created during preprocessing and
    should have column names in the form of 't-*lag number*' (e.g. 't-1').
    """

    def __init__(self, lag=1):
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X.iloc[:, -self.lag]


class MovingAverage(BaseEstimator, RegressorMixin):
    """
    Moving Average estimator which returns the average of lag t-1 up untill t-*window size*.
    The lags should be columns that are created during preprocessing and
    should have column names in the form of 't-*lag number*' (e.g. 't-1').
    """

    def __init__(self, window=3):
        self.window = window

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X.iloc[:, -self.window :].mean(axis=1)


class Zero(BaseEstimator, RegressorMixin):
    """
    Zero estimator that returns a zero forecast.
    """

    def __init__(self, window=3):
        self.window = window

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((X.shape[0], 1))
