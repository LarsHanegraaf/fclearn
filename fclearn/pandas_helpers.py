"""Helper functions for Pandas DataFrames with multiple time series."""

import numpy as np
import pandas as pd


def cumcount(series: pd.Series, count_if: str, value) -> pd.Series:
    """Cumulative counts subsequent occurrences.

    Counts if True, resets when false

    Args:
        series (pd.Series): Series to evaluate
        count_if (str): either 'eq' -> equals or 'ne' -> not equals
        value (undefined): Value to compare

    Returns:
        pd.Series

    """
    if count_if == "eq":
        to_count = (series == value).astype("int")
    if count_if == "ne":
        to_count = (series != value).astype("int")

    reset_column = ~to_count.astype("int")
    reset_column = reset_column.diff()
    reset_column.fillna(0, inplace=True)
    reset_column.replace(to_replace=-1, value=0, inplace=True)

    to_count = to_count.cumsum()
    to_subtract = to_count * reset_column
    to_subtract = to_subtract.cummax()
    return (to_count - to_subtract).astype("int64")


def cumsum(series_to_sum: pd.Series, series_to_index: pd.Series, cumsum_if: str, value):
    """Cumulative sum subsequent equal values.

    Sums if expression evaluates to True, resets the sum if it is False.

    Args:
        series_to_sum (pd.Series): Series to evaluate
        series_to_index (pd.Series): Series to index
        cumsum_if (str): 'eq' -> equals or 'ne' -> not equals
        value (undefined): Value to compare

    Returns:
        pd.Series

    """
    if cumsum_if == "eq":
        to_count = (series_to_index == value).astype("int")
    if cumsum_if == "ne":
        to_count = (series_to_index != value).astype("int")

    reset_column = ~to_count.astype("int")
    reset_column = reset_column.diff()
    reset_column.fillna(0, inplace=True)
    reset_column.replace(to_replace=-1, value=0, inplace=True)

    to_sum = to_count.copy()
    to_sum[to_count.astype("bool")] = series_to_sum[to_count.astype("bool")]
    to_sum = to_sum.cumsum()
    to_subtract = to_sum * reset_column
    to_subtract = to_subtract.cummax()

    return to_sum - to_subtract


def unnesting(df: pd.DataFrame, explode: list, axis: int) -> pd.DataFrame:
    """Unnests Pandas DataFrame columns.

    Args:
        df (pd.DataFrame): DataFrame to expand
        explode (list): list of strings with the names of the columns
        axis (int): axis to expand to

    Returns:
        pd.DataFrame

    """
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat(
            [pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1
        )
        df1.index = idx

        return df1.join(df.drop(explode, 1), how="left")
    else:
        df1 = pd.concat(
            [
                pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x)
                for x in explode
            ],
            axis=1,
        )
        return df1.join(df.drop(explode, 1), how="left")


def get_time_series_combinations(df: pd.DataFrame, groupby: list) -> list:
    """Gets all combinations of the groupby and returns them in a list.

    Args:
        df (pd.DataFrame): DataFrame with MultiIndex
        groupby (list): list of index names that are in the multiindex

    Returns:
        list

    """
    # TODO make it working for more than to items in the groupby
    skuid = df.index.get_level_values(groupby[0])
    customer = df.index.get_level_values(groupby[1])

    result = np.array(list(zip(skuid, customer)))

    result = np.unique(result, axis=0)

    result = [tuple(x) for x in result]

    return result


def get_series(df: pd.DataFrame, index_tuple: tuple) -> pd.DataFrame:
    """Return the DataFrame of one DFU based on a tuple.

    Args:
        df (pd.DataFrame): DataFrame to slice
        index_tuple (tuple): tuple with the series to return ('SKUID, 'ForecastGroupID')

    Returns:
        pd.DataFrame

    """
    return df.loc[
        (df.index.get_level_values("SKUID") == index_tuple[0])
        & (df.index.get_level_values("ForecastGroupID") == index_tuple[1])
    ]
