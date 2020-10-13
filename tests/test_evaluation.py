import numpy as np

from fclearn.evaluation import mape


class TestMape:
    def test_one(self):
        result = mape(np.array([2]), np.array([1]))
        assert result == np.array([1])

    def test_two(self):
        result = mape(np.array([1]), np.array([2]))
        assert result == np.array([0.5])

    def test_three(self):
        result = mape(np.array([1]), np.array([1]))
        assert result == np.array([0])

    def test_four(self):
        result = mape(np.array([4]), np.array([1]))
        assert result == np.array([1])
