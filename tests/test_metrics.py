import numpy as np
import pytest

# metrics.py imports keras at module level, which requires tensorflow.
# Use importorskip to gracefully skip if keras/tf aren't available.
metrics = pytest.importorskip("mlgear.metrics", reason="requires keras/tensorflow")

rmse = metrics.rmse
crps_score = metrics.crps_score
crps_score_ = metrics.crps_score_


class TestRmse:
    def test_perfect_prediction(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self):
        assert rmse([0, 0], [1, 1]) == pytest.approx(1.0)

    def test_single_value(self):
        assert rmse([3], [5]) == pytest.approx(2.0)


class TestCrpsScore:
    def test_perfect_prediction(self):
        actual = np.zeros((2, 199))
        result = crps_score(actual, actual)
        assert result == pytest.approx(0.0)

    def test_nonzero(self):
        actual = np.zeros((1, 199))
        predicted = np.ones((1, 199))
        result = crps_score(actual, predicted)
        assert result > 0


class TestCrpsScore_:
    def test_perfect(self):
        actual = np.zeros((2, 199))
        result = crps_score_(actual, actual)
        assert result == pytest.approx(0.0)
