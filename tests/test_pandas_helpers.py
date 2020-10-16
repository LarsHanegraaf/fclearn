"""Tests for the pandas_helpers module."""
import pandas as pd

from fclearn.pandas_helpers import cumcount, cumsum


class TestCumcount:
    """Test cumcount."""

    def test_one(self):
        """Cumcount works for 0 and 1 values in 'eq' mode."""
        series = pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        result = cumcount(series, "eq", 1)
        pd.testing.assert_series_equal(
            pd.Series([0, 1, 2, 3, 4, 0, 1, 0, 0, 0, 1, 2]), result
        )

    def test_two(self):
        """Cumcount works for 0 and 1 values in 'ne' mode."""
        series = pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        result = cumcount(series, "ne", 1)
        pd.testing.assert_series_equal(
            pd.Series([1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0]), result
        )


class TestCumsum:
    """Test cumsum."""

    def test_one(self):
        """Cumsum works for 0 and 1 values in 'eq' mode."""
        to_evaluate = pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        to_sum = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        result = cumsum(to_sum, to_evaluate, "eq", 1)
        pd.testing.assert_series_equal(
            pd.Series([0, 1, 3, 6, 10, 0, 6, 0, 0, 0, 10, 21], dtype="float"), result
        )

    def test_two(self):
        """Cumcsum works for 0 and 1 values in 'ne' mode."""
        to_evaluate = pd.Series([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        to_sum = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        result = cumsum(to_sum, to_evaluate, "ne", 1)
        pd.testing.assert_series_equal(
            pd.Series([0, 0, 0, 0, 0, 5, 0, 7, 15, 24, 0, 0], dtype="float"), result
        )
