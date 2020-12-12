"""Decomposition methods as a Scikit-learn transformer."""
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import STL


class DecomposeTransformer(BaseEstimator, TransformerMixin):
    """Decomposition transformer for classical and STL Decomposition."""

    def __init__(self, method):
        """Constructor."""
        self.method = method

    def fit(self, X, y=None):
        """Fit the transformer."""
        if self.method == "stl":
            self.decompose_results = STL(X.values, period=52, seasonal=13).fit()
        if self.method == "classic":
            self.decompose_results = sm.tsa.seasonal_decompose(
                X.values, model="additive", period=52, extrapolate_trend=True
            )
        return self

    def transform(self, X):
        """Transform the series into trend, seasonal and remainder."""
        X_ = pd.DataFrame(index=X.index)
        X_["trend"] = self.decompose_results.trend
        X_["seasonal"] = self.decompose_results.seasonal
        X_["error"] = self.decompose_results.resid
        return X_
