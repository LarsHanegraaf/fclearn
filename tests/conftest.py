"""File with test fixtures."""

import numpy as np
import pandas as pd
import pytest


def demand_factory(number_of_skus, number_of_customers, number_of_days):
    """Factory function for creating a test DataFrame."""
    rows = []
    for sku in range(number_of_skus):
        for customer in range(number_of_customers):
            for day in range(number_of_days):
                date = pd.to_datetime("2017-01-02") + np.timedelta64(day, "D")
                # Create a cycle where demand is steady, but is different for every
                # customer. Last column is a 'predictor' that corresponds with the
                # day of the week.
                rows.append([sku, customer, date, 10 + customer * 10, date.weekday])
    return rows


@pytest.fixture
def demand_df():
    """Dataframe with fake demand."""
    df = pd.DataFrame(
        data=demand_factory(1, 2, 28),
        columns=["SKUID", "ForecastGroupID", "Date", "HL_sum", "Predictor"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(["SKUID", "ForecastGroupID", "Date"])
    return df
