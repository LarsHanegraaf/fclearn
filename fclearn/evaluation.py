"""Plotting and scoring fuctions for dataframes with multiple time series."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fclearn import pandas_helpers


def create_prediction_df(X_test, y_test, estimator, name, for_writing_to_db=False):
    """Creates a dataframe for evaluation and writing to a database.

    Args:
        X_test (pd.DataFrame): Testing DataFrame with predictors
        y_test (pd.DataFrame): Testing DataFrame with targets
        estimator (sklearn.BaseEstimator): Estimator to predict with
        name (string): Name of the estimator, used for column name
        for_writing_to_db (bool): True if it should be formatted for writing to the DB.

    Returns:
        pd.DataFrame: DataFrame with the predictions.

    """
    predictions = estimator.predict(X_test)
    if for_writing_to_db:
        y_pred = pd.DataFrame(index=y_test.index, columns=["HL", "Algorithm"])
        y_pred["HL"] = predictions
        y_pred["Algorithm"] = name
        y_pred.reset_index(inplace=True)
    else:
        y_pred = pd.DataFrame(index=y_test.index, columns=[name])
        y_pred[name] = predictions
    return y_pred


def plot_series(
    df: pd.DataFrame, sku_dict: dict, fcp_dict: dict, x: str = None, y: str = None
) -> None:
    """Creates an individual plot for every time series.

    Args:
        df (pd.DataFrame): DataFrame to plot
        sku_dict (dict): Dictionary with the SKUID as key and the name as value
        fcp_dict (dict): Dictionary with the FCP as key and the name as value
        x (str): column name of x axis
        y (str): column name of y axis
    """
    groupby = ["SKUID", "ForecastGroupID"]
    for index in pandas_helpers.get_time_series_combinations(df, groupby):
        data = pandas_helpers.get_series(df, index)
        sku = sku_dict[index[0]]
        forecastgroup = fcp_dict[index[1]]
        data.index = data.index.droplevel(["SKUID", "ForecastGroupID"])
        plt.figure(figsize=(20, 5))
        sns.lineplot(data=data, x=x, y=y).set_title(
            "{} - {}".format(sku, forecastgroup)
        )
        plt.show()


def mape(forecast: np.array, actual: np.array) -> np.array:
    """Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        forecast (np.array): Array with the forecasted values
        actual (np.array): Array with the actual values

    Returns:
        np.array

    """
    ATOL = 1e-5
    # ignore numpy divide error
    with np.errstate(divide="ignore"):
        res = np.abs(actual - forecast) / actual
    res[actual <= 0] = 1  # error is 100%
    res[
        np.isclose(actual, forecast, atol=ATOL)
    ] = 0  # error is zero. This line should be last.

    # If the actual or forecast is np.nan always return np.nan
    res[np.logical_or(np.isnan(actual), np.isnan(forecast))] = np.nan

    res = np.clip(res, 0, 1)

    return res
