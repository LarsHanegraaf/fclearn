"""Tests for model_selection.py."""

import numpy as np
import pandas as pd
import pytest

from fclearn.model_selection import create_rolling_forward_indices, train_test_split

groupby = ["SKUID", "ForecastGroupID"]


class TestTrainTestSplit:
    """Test train_test_split()."""

    def test_one(self, demand_df):
        """Whether splits on date."""
        X = pd.DataFrame(
            data=[["2017-01-02", 1], ["2017-01-03", 2]], columns=["Date", "value"]
        )
        y = pd.DataFrame(
            data=[["2017-01-02", 3], ["2017-01-03", 4]], columns=["Date", "value"]
        )
        X["Date"] = pd.to_datetime(X["Date"])
        y["Date"] = pd.to_datetime(y["Date"])
        X.set_index("Date", inplace=True)
        y.set_index("Date", inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, "2017-01-03")

        # X_train
        pd.testing.assert_frame_equal(
            X_train,
            pd.DataFrame(
                data=[1],
                columns=["value"],
                index=pd.Index([pd.to_datetime("2017-01-02")], name="Date"),
            ),
        )

        # X_test
        pd.testing.assert_frame_equal(
            X_test,
            pd.DataFrame(
                data=[2],
                columns=["value"],
                index=pd.Index([pd.to_datetime("2017-01-03")], name="Date"),
            ),
        )

        # y_train
        pd.testing.assert_frame_equal(
            y_train,
            pd.DataFrame(
                data=[3],
                columns=["value"],
                index=pd.Index([pd.to_datetime("2017-01-02")], name="Date"),
            ),
        )

        # y_test
        pd.testing.assert_frame_equal(
            y_test,
            pd.DataFrame(
                data=[4],
                columns=["value"],
                index=pd.Index([pd.to_datetime("2017-01-03")], name="Date"),
            ),
        )


class TestRollingCV:
    """Test create_rolling_forward_indices()."""

    def test_start_warning(self, demand_df):
        """Gives warning when start date is not monday."""
        pytest.warns(
            Warning,
            create_rolling_forward_indices,
            demand_df,
            groupby,
            "2017-01-17",
            "2017-01-22",
            7,
            7,
            7,
        )

    def test_end_warning(self, demand_df):
        """Gives warning when end date is not sunday."""
        pytest.warns(
            Warning,
            create_rolling_forward_indices,
            demand_df,
            groupby,
            "2017-01-16",
            "2017-01-23",
            7,
            7,
            7,
        )

    def test_one(self, demand_df):
        """Two folds are received when start and stop are 13 days apart."""
        start_date = pd.to_datetime("2017-01-16")
        cv = create_rolling_forward_indices(
            demand_df,
            groupby,
            start_date,
            start_date + np.timedelta64(13, "D"),
            7,
            7,
            7,
        )
        assert len(cv) == 2

    def test_two(self, demand_df):
        """Gap between the last item train set and first item test is 7 days."""
        start_date = pd.to_datetime("2017-01-16")
        cv = create_rolling_forward_indices(
            demand_df,
            groupby,
            start_date,
            start_date + np.timedelta64(13, "D"),
            7,
            7,
            7,
        )
        number_of_series = 2
        for train, test in cv:
            for serie in range(number_of_series):
                # Find the length of every inidivual serie
                serie_length_train = int(len(train) / number_of_series)
                serie_length_test = int(len(test) / number_of_series)

                # Find the last index of every serie in the train set
                highest_index_train = (
                    serie_length_train - 1 + serie_length_train * serie
                )

                # Find the last index of every serie in the test set
                highest_index_test = serie_length_test - 1 + serie_length_test * serie

                assert test[highest_index_test] - train[highest_index_train] == 7

    def test_start_day_test_is_same(self, demand_df):
        """Start day of test set is the same as train set."""
        start_date = pd.to_datetime("2017-01-16")
        cv = create_rolling_forward_indices(
            demand_df,
            groupby,
            start_date,
            start_date + np.timedelta64(13, "D"),
            7,
            7,
            7,
        )
        number_of_series = 2
        for train, test in cv:
            for serie in range(number_of_series):
                # Find the length of every inidivual serie
                serie_length_train = int(len(train) / number_of_series)
                serie_length_test = int(len(test) / number_of_series)

                # Pick the first index of the train set
                first_index_train_in_fold = 0 + serie_length_train * serie

                # Pick the first index of the test set
                first_index_test_in_fold = 0 + serie_length_test * serie

                # Pick the index from the train and test indices
                first_index_train = train[first_index_train_in_fold]
                first_index_test = test[first_index_test_in_fold]

                # As iloc returns one row, the level of the MultiIndex is stored
                # the 'name' attribute of the series.
                # TODO make sure this test works when 'Date' is not at level 3.
                print(first_index_train, train, first_index_test, test)
                assert (
                    demand_df.iloc[first_index_train].name[2].weekday()
                    == demand_df.iloc[first_index_test].name[2].weekday()
                )

    def test_start_day_is_same_as_parameter(self, demand_df):
        """Start day of test set is the same as parameter."""
        start_date = pd.to_datetime("2017-01-16")
        cv = create_rolling_forward_indices(
            demand_df,
            groupby,
            start_date,
            start_date + np.timedelta64(13, "D"),
            7,
            7,
            7,
        )
        number_of_series = 2

        train, test = cv[0]
        for serie in range(number_of_series):
            # Find the length of every inidivual serie
            serie_length_test = int(len(test) / number_of_series)

            # Pick the first index of the test set
            first_index_test_in_fold = 0 + serie_length_test * serie

            # Pick the index from the train and test indices
            first_index_test = test[first_index_test_in_fold]

            print(demand_df.iloc[test])

            assert start_date == demand_df.iloc[first_index_test].name[2]
