"""Tests for model_selection.py."""

import pandas as pd

from fclearn.model_selection import train_test_split


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
