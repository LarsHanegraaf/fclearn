import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from fclearn import pandas_helpers


def plot_series(df: pd.DataFrame, sku_dict: dict, fcp_dict: dict, x: str = None, y: str = None) -> None:
    """
    Plot all the time series in an individual plot to inspect the results.

    :param df: DataFrame to plot
    :param groupby: Keys that specify the DFU
    :param columns: Columns to plot
    :return: None
    """
    groupby = ['SKUID', 'ForecastGroupID']
    for index in pandas_helpers.get_time_series_combinations(df, groupby):
        data = pandas_helpers.get_series(df, index)
        sku = sku_dict[index[0]]
        forecastgroup = fcp_dict[index[1]]
        data.index = data.index.droplevel(['SKUID', 'ForecastGroupID'])
        plt.figure(figsize=(20, 5))
        sns.lineplot(data=data, x=x, y=y).set_title("{} - {}".format(sku, forecastgroup))
        plt.show()

def mape(forecast, actual):
    """
    Calculates the Mean Absolute Percentage Error.
    Forecast and actual should be Numpy arrays
    
    Error is clipped between 0 and 1
    """
    ATOL = 1
    # ignore numpy divide error
    with np.errstate(divide='ignore'):
        res = np.abs(actual - forecast) / actual
    res[actual <= 0] = 1  # error is 100%
    res[np.isclose(actual, forecast, atol=ATOL)] = 0  # error is zero. This line should be last.

    # If the actual or forecast is np.nan always return np.nan
    res[np.logical_or(np.isnan(actual), np.isnan(forecast))] = np.nan

    res = np.clip(res, 0, 1)
    
    return res