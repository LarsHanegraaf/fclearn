"""Decomposition methods as a Scikit-learn transformer."""
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import STL


class STLTransformer(BaseEstimator, TransformerMixin):
    """STL Decomposition."""

    def __init__(self):
        """Constructor."""
        pass

    def fit(self, X, y=None):
        """Fit the transformer."""
        self.stl_results = STL(X).fit()
        return self

    def transform(self, X):
        """Transform the series into trend, seasonal and remainder."""
        return X
