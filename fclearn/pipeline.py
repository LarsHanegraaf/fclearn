"""Extensions of the scikit-learn Pipelines, so they can be used with Pandas."""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class PandasFeatureUnion(FeatureUnion):
    """A Pandas implementation of a sklearn FeatureUnion.

    It returns a Pandas Dataframe.
    Implemented because of the lack of column names and indices of the sklearn
    implementation.

    Taken from
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html

    Updated with the source code of sklearn to work with version 0.23
    """

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform the Feature Union."""
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            # Next line is different from source code
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        """Helper function for concatenating all the dataframes."""
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        """Transform all items to a Feature Union."""
        for _, t in self.transformer_list:
            # TODO: Remove in 0.24 when None is removed
            if t is None:
                warnings.warn(
                    "Using None as a transformer is deprecated "
                    "in version 0.22 and will be removed in "
                    "version 0.24. Please use 'drop' instead.",
                    FutureWarning,
                )
                continue
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            # Next line is different from source code
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
