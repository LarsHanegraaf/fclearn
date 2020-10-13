"""Tests for the evaluation module."""
import numpy as np

from fclearn.evaluation import mape


class TestMape:
    """Description of TestMape."""

    def test_one(self):
        """Test forecast 2, actual 1, error of 100%."""
        result = mape(np.array([2]), np.array([1]))
        assert result == np.array([1])

    def test_two(self):
        """Test forecast 1, actual 2, error of 50%."""
        result = mape(np.array([1]), np.array([2]))
        assert result == np.array([0.5])

    def test_three(self):
        """Test forecast 1, actual 1, error of 0%."""
        result = mape(np.array([1]), np.array([1]))
        assert result == np.array([0])

    def test_four(self):
        """Test forecast 4, actual 1, error of 100% due to clipping."""
        result = mape(np.array([4]), np.array([1]))
        assert result == np.array([1])
