"""Helper function for reconciling hierarchical series."""

import pandas as pd


def create_reconciliation_matrix(
    df, groupby, features_to_reconcile, groupby_mapping=None
):
    """Creates a reconciliation matrix for the base series.

    Args:
        df (pd.DataFrame): DataFrame that should contain the DFU and features that the
            reconciliation should be based on.
        groupby (list): groupby that specifies the DFU (Demand Forecasting Unit).
        features_to_reconcile (list): list of features that should be included in the
            output matrix.
        groupby_mapping (list): list of dictionaries. If the groupby key should have a
            different mapping for the 'base series' provide a dictionary that should be
            used for renaming.

    Returns:
        pd.DataFrame: **reconciliation_matrix** - DataFrame with the matrix.
    """
    df_ = df.reset_index()[groupby + features_to_reconcile]
    df_.set_index(groupby, inplace=True)

    # Filter for unique values for the index
    df_ = df_.loc[~df_.index.duplicated()]

    # Add reconciliation column for all series
    df_["All series"] = "all"

    # Add reconciliation column for base series
    base = df_.reset_index()[groupby]

    # Rename base series columns when needed
    if groupby_mapping is not None:
        assert len(groupby) == len(groupby_mapping)
        for index in range(len(groupby_mapping)):
            base.iloc[:, index] = base.iloc[:, index].replace(groupby_mapping[index])

    # Create a name for the base series
    df_["Base series"] = (
        base.astype("str").apply(lambda x: " - ".join(x), axis=1).values
    )

    return df_


def reconcile_column(base_series, reconciliation_matrix, level_name, target_column):
    """Reconcile a single column of the reconciliation matrix.

    Reconciles a single column.

    Args:
        base_series (pd.DataFrame): Unstacked DataFrame with the DFU as MultiIndex and
            the dates as columns.
        reconciliation_matrix (pd.DataFrame): DataFrame with the reconciliation values
            for each DFU.
        level_name (string): Column name of the reconciliation_matrix that should be
            used for reconciliation.
        target_column (string): Name of the original target column (used for creating
            the proper output).

    Returns:
        pd.DataFrame: **reconciled_base_series** - DataFrame with series reconciled by
            column 'level_name'.
    """
    df_ = base_series.copy()

    # Add column name to the level to reconcile
    rm = reconciliation_matrix[[level_name]]
    rm = "{} - ".format(rm.columns[0]) + rm

    # Add reconcile information
    df_ = df_.join(rm)
    df_ = df_.groupby(level_name).sum()

    # Rename the index
    df_.index.name = "DFU"

    df_ = pd.DataFrame(df_.stack(), columns=[target_column])
    return df_


def reconcile_dataframe(base_series, reconciliation_matrix, target_column):
    """Reconcile all variants that are in the reconciliation matrix.

    Creates a DataFrame with all the possible reconciliation options.

    Args:
        base_series (pd.DataFrame): The lowest level of possible series in a stacked
            format: DFU + Date as MultiIndex
        reconciliation_matrix (pd.DataFrame): DataFrame with the reconciliation values
            for each DFU.
        target_column: Name of the column to reconcile of the base_series.

    Returns:
        pd.DataFrame: **reconciled_base_series** - DataFrame with series reconciled for
            all possible columns in the reconiliation_matrix.
    """
    dfs = []
    # Unstack in order to work with reconcile_column
    base_series_ = base_series[target_column].unstack()
    # Loop over every reconciliation column
    for column in reconciliation_matrix:
        dfs.append(
            reconcile_column(base_series_, reconciliation_matrix, column, target_column)
        )
    df_ = pd.concat(dfs)
    return df_
