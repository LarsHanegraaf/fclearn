import pandas as pd
import numpy as np


def cumcount(series: pd.Series, count_if: str, value) -> pd.Series:
    """
    Cumulative counts subsequent occurrences if the expression evaluate True.
    Resets the count if it is False.

    :param series: Series to evaluate
    :param count_if: either 'eq' -> equals or 'ne' -> not equals
    :param value: Value to compare
    :return: cumcount series: the series with the cumulative count
    """
    if count_if == 'eq':
        to_count = (series == value).astype('int')
    if count_if == 'ne':
        to_count = (series != value).astype('int')

    reset_column = ~to_count.astype('int')
    reset_column = reset_column.diff()
    reset_column.fillna(0, inplace=True)
    reset_column.replace(to_replace=-1, value=0, inplace=True)

    to_count = to_count.cumsum()
    to_subtract = to_count * reset_column
    to_subtract = to_subtract.cummax()
    return to_count - to_subtract


def cumsum(series_to_sum: pd.Series, series_to_index: pd.Series, cumsum_if: str, value):
    """
    Cumulative sum subsequent occurrences if the expression evaluate True.
    Resets the sum if it is False.

    :param series_to_sum: Series to evaluate
    :param series_to_index: Series to index
    :param cumsum_if: either 'eq' -> equals or 'ne' -> not equals
    :param value: Value to compare
    :return: cumcount series: the series with the cumulative count
    """
    if cumsum_if == 'eq':
        to_count = (series_to_index == value).astype('int')
    if cumsum_if == 'ne':
        to_count = (series_to_index != value).astype('int')

    reset_column = ~to_count.astype('int')
    reset_column = reset_column.diff()
    reset_column.fillna(0, inplace=True)
    reset_column.replace(to_replace=-1, value=0, inplace=True)

    to_sum = to_count.copy()
    to_sum[to_count.astype('bool')] = series_to_sum[to_count.astype('bool')]
    to_sum = to_sum.cumsum()
    to_subtract = to_sum * reset_column
    to_subtract = to_subtract.cummax()

    return to_sum - to_subtract


def unnesting(df: pd.DataFrame, explode: list, axis: int) -> pd.DataFrame:
    """
    Unnests dataframe columns.
    In this case used to transform a Data Range that is in cell, to rows

    :param df: DataFrame to expand
    :param explode: list of strings with the names of the columns
    :param axis: axis to expand to
    :return: df: Unnested dataframe
    """
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx

        return df1.join(df.drop(explode, 1), how='left')
    else:
        df1 = pd.concat([
            pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='left')


def get_time_series_combinations(df: pd.DataFrame, groupby: list) -> list:
    """
    Gets all combinations of the groupby and returns them in a list

    :param df: DataFrame with MultiIndex
    :param groupby: list of index names that are in the multiindex
    :return: list: of combinations in the index
    """
    # TODO make it working for more than to items in the groupby
    skuid = df.index.get_level_values(groupby[0])
    customer = df.index.get_level_values(groupby[1])

    result = np.array(list(zip(skuid, customer)))

    result = np.unique(result, axis=0)

    result = [tuple(x) for x in result]

    return result


def get_series(df: pd.DataFrame, index_tuple: tuple) -> pd.DataFrame:
    """
    Return the DataFrame of one DFU bases on a tuple received from e.g. get_time_series_combinations().

    :param df: DataFrame to look up in
    :param index_tuple: tuple with the series to return ('SKUID, 'ForecastGroupID')
    :return:
    """
    return df.loc[(df.index.get_level_values('SKUID') == index_tuple[0]) & (
                df.index.get_level_values('ForecastGroupID') == index_tuple[1])]
