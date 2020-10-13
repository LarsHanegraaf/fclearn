import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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