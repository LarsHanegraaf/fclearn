"""Imputation fucntions for missing data for time series."""

import numpy as np
import pandas as pd

from fclearn import pandas_helpers


def extrapolate_promotions(
    rows_to_extrapolate: pd.DataFrame,
    groupby: list,
    min_promo_duration: int = 7,
    min_load_in_duration: int = 13,
) -> pd.DataFrame:
    """Extrapolates promotions of the demand dataframe using standard parameters.

    Returns the extrapolated DataFrame.

    Args:
        rows_to_extrapolate (pd.DataFrame): Dataframe with the rows that have missing
            promo information
        groupby (list): parameter used to group DFU's
        min_promo_duration (int): The duration of the promotional period that should be
            used. This duration is placed on the end of the load in window
        min_load_in_duration (int): Minimum length of every load in that gets
            extrapolated.

    Returns:
        pd.DataFrame

    """
    df = rows_to_extrapolate.copy()

    df.sort_values(by=["SKUID", "ForecastGroupID", "Date"], inplace=True)

    df["datediff"] = df.groupby(groupby).Date.apply(lambda x: x.diff())

    # Create a group id for items that are within a range of 7 days of each other
    df["is_new_group"] = (df["datediff"] / np.timedelta64(1, "D") > 7).astype("int")
    # Every first item of DFU is also a new group (can be filtered using isna due to
    # diff() two lines before),
    df.loc[df["datediff"].isna(), "is_new_group"] = 1
    df["id"] = df["is_new_group"].cumsum()

    df = df.groupby(["SKUID", "ForecastGroupID", "id"]).agg({"Date": ["min", "max"]})

    # Drop 'Date' level from MultiIndex column
    df.columns = df.columns.droplevel(0)

    # Create a column for the weekstart of the date under investigation
    df["week_start"] = df["min"] - np.array(
        [np.timedelta64(value, "D") for value in df["min"].dt.weekday]
    )

    df["week_end"] = df["max"] + np.array(
        [np.timedelta64(6 - value, "D") for value in df["max"].dt.weekday]
    )

    # Calculate duration of period
    df["duration"] = (df["week_end"] - df["week_start"]) / np.timedelta64(1, "D")

    # As determined in the EDA, the duration of an 'inlading' is often the same for
    # multiple retailers
    # Thus we want to make sure when we find durations of e.g. one week, they are
    # changed to the min_load_in_duration
    df["duration"] = df["duration"].apply(
        lambda x: min_load_in_duration if x < min_load_in_duration else x
    )

    # Calculate the start date. The promotion is starting at least min_promo_duration
    # days from the end of the load in period.
    df["start_date_promo"] = df.apply(
        lambda x: x["week_end"]
        - np.timedelta64(int(x["duration"]) - min_promo_duration, "D"),
        axis=1,
    )
    df["start_date_load_in"] = df.apply(
        lambda x: x["week_end"] - np.timedelta64(int(x["duration"]), "D"), axis=1
    )

    # Create a date range from the start to the load in untill the end
    df["Date"] = df.apply(
        lambda x: pd.date_range(x["start_date_load_in"], x["week_end"]), axis=1
    )

    # Expand the date range into rows
    df = pandas_helpers.unnesting(df, ["Date"], 1)

    # Create 'actie' and 'inlading' column
    df["actie"] = (df["Date"] >= df["start_date_promo"]).astype("int")
    df["inlading"] = 1

    # Drop unnecessary created index of pandas_helpers.unnesting()
    df.index = df.index.droplevel(2)

    return (
        df[["Date", "actie", "inlading"]]
        .reset_index()
        .set_index(["SKUID", "ForecastGroupID", "Date"])
    )
