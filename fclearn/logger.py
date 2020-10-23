"""Extra classes for Bayesion optimization."""
import numpy as np
import pandas as pd
from bayes_opt import Events
from bayes_opt.observer import _Tracker
from sklearn.preprocessing import PowerTransformer


class PandasLogger(_Tracker):
    """Logger for bayesian-optimization library."""

    def __init__(self):
        """Set the output dataframe."""
        self.data = pd.DataFrame(
            columns=[
                "mean_test_rmse",
                "param_regression__regressor__max_depth",
                "param_regression__regressor__n_estimators",
                "param_regression__transformer",
            ]
        )
        super(PandasLogger, self).__init__()

    def update(self, event, instance):
        """Observer method that adds a row for every iteration to the DF."""
        if event == Events.OPTIMIZATION_STEP:
            result = instance.res[-1]
            row = {
                "mean_test_rmse": result["target"],
                "param_regression__regressor__max_depth": int(
                    result["params"]["max_depth"]
                ),
                "param_regression__regressor__n_estimators": int(
                    result["params"]["n_estimators"]
                ),
                "param_regression__transformer": "None"
                if result["params"]["transformer"] < 0.5
                else PowerTransformer(),
            }
            self.data = self.data.append(row, ignore_index=True)
        if event == Events.OPTIMIZATION_END:
            self.data["rank_test_rmse"] = self.data.sort_values(
                "mean_test_rmse", ascending=False
            )[["mean_test_rmse"]].apply(
                lambda x: pd.Series(np.arange(len(x)) + 1, x.index)
            )

        self._update_tracker(event, instance)
