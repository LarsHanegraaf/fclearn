from fclearn import evaluation
import numpy as np

def test_mape():
    result = evaluation.mape(np.array([1]), np.array([1]))
    assert result == 0