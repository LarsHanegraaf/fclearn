"""Classes and functions used for model selection.

These classes and methods are extensions to be used with sklearn.
"""

import numpy as np
import pandas as pd


def train_test_split(
    X: pd.DataFrame, y: pd.DataFrame, split_date: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Train-Test split for time series data.

    Args:
        X (pd.DataFrame): DataFrame with the predictors,
            should have a MultiIndex with a column Date
        y (pd.DataFrame): DataFrame with the target,
            should have a MultiIndex with a column Date
        split_date (string): Date on which the test set should start

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_date = pd.to_datetime(split_date)
    X_train = X.loc[X.index.get_level_values("Date") < split_date]
    X_test = X.loc[X.index.get_level_values("Date") >= split_date]
    y_train = y.loc[y.index.get_level_values("Date") < split_date]
    y_test = y.loc[y.index.get_level_values("Date") >= split_date]
    return X_train, X_test, y_train, y_test


def create_rolling_forward_indices(
    df,
    groupby,
    start_date,
    end_date,
    retrain_interval,
    gap,
    data_augmentation_factor=None,
):
    """Creates a sklearn CV iterator object for rolling forward validation.

    Creates indices for a train and test split that can be used in a
    GridSearchCV (or a rolling forecast predictor).The result is a
    'cross-validation' iterable, where in every fold the train set is
    increasing and the test set only contains one step ahead.

    Args:
        df (pd.DataFrame): The time series to be iterated on. Should be sorted
            on DFU and Date!
        groupby (list): List of the groupby that specifies a DFU
        start_date (string): Start date of the trian set (preferably a Monday)
        end_date (string): End date of the validation set?
        retrain_interval (int): Number of days that are in a fold, the higher
            this number, the lesser the amount of folds
        gap (int): Number of days between the end of the train set and start
            of test et
        data_augmentation_factor (int): When data augmentation is used (e.g.
            training on 7-days ahead predictions on daily intervals), make sure
            that the gap has sufficient value to make sure no future leakage
            happens.

    Raises:
        NotImplementedError: When data_augmentation_factor is None
        ValueError: When the dates inputted are not OK

    Returns:
        list: Indices to be used in a GridSearchCV or Cross Validation
    """
    start_date_ = pd.to_datetime(start_date)
    end_date_ = pd.to_datetime(end_date)

    # Value checking
    if df.index.get_level_values("Date").min() > pd.to_datetime(start_date_):
        raise ValueError("Start date should be greater than the start date of the df")

    if df.index.get_level_values("Date").max() < pd.to_datetime(end_date_):
        raise ValueError("End date should be smaller than the end date of the df")

    if start_date >= end_date:
        raise ValueError("Start date shoud be before the end date")

    # Find serie shapes
    number_of_series = df.groupby(groupby).count().shape[0]
    number_of_observations_per_serie = df.groupby(groupby).count().mean()[0]

    if not number_of_observations_per_serie.is_integer():
        raise ValueError("The series in the DataFrame do not have the same length")
    else:
        number_of_observations_per_serie = int(number_of_observations_per_serie)

    # Create a serie with all dates within the range
    indices = df.index.get_level_values("Date")
    indices = pd.Series(indices)
    indices = indices.loc[(indices >= start_date_) & (indices <= end_date_)]
    indices = indices.unique()

    # Find the index of the start_date argument on the first time serie
    start_date_index = df.index.get_level_values("Date").get_indexer_for([start_date_])[
        0
    ]

    if start_date_index == -1:
        raise ValueError("start_date is not in index")

    # Calculate all relative start and stop indices for the train and test split
    # (Relative to the start_date)

    if data_augmentation_factor is not None:
        folds = []

        num_folds = (
            int((end_date_ - start_date_) / np.timedelta64(1, "D")) // retrain_interval
        ) + 1

        # Define test size of every fold
        test_size = retrain_interval / data_augmentation_factor
        if not test_size.is_integer():
            raise ValueError(
                "Retrain interval should be a multiple of the data augmentation factor"
            )
        test_size = int(test_size)

        for fold in range(num_folds):
            trains = []
            tests = []

            for serie in range(number_of_series):

                multiplier = serie * number_of_observations_per_serie

                train = list(
                    range(
                        0 + multiplier,
                        (start_date_index + multiplier)
                        + (fold * retrain_interval)
                        - gap,
                    )
                )
                test = list(
                    range(
                        start_date_index + multiplier + (fold * retrain_interval),
                        start_date_index
                        + multiplier
                        + (fold * retrain_interval)
                        + test_size,
                    )
                )

                trains.append(train)
                tests.append(test)

            trains = np.array(trains).flatten()
            tests = np.array(tests).flatten()
            folds.append((trains, tests))

    else:
        raise NotImplementedError("Not implemented for no data augmentation")

    return folds
