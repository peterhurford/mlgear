import numpy as np
import pytest

from mlgear.aggregators import mean_diff, std_diff, min_diff, max_diff, nth_smallest, second_min


class TestMeanDiff:
    def test_uniform_spacing(self):
        assert mean_diff([1, 2, 3, 4, 5]) == 1.0

    def test_uneven_spacing(self):
        result = mean_diff([1, 3, 6])
        assert result == pytest.approx(2.5)

    def test_unsorted_input(self):
        assert mean_diff([5, 1, 3]) == pytest.approx(2.0)

    def test_negative_values(self):
        assert mean_diff([-3, -1, 1]) == pytest.approx(2.0)

    def test_floats(self):
        result = mean_diff([0.1, 0.4, 0.9])
        assert result == pytest.approx(0.4)


class TestStdDiff:
    def test_uniform_spacing_is_zero(self):
        assert std_diff([1, 2, 3, 4, 5]) == pytest.approx(0.0)

    def test_uneven_spacing(self):
        result = std_diff([1, 3, 6])
        assert result > 0


class TestMinDiff:
    def test_uniform(self):
        assert min_diff([1, 2, 3, 4]) == 1.0

    def test_uneven(self):
        assert min_diff([1, 2, 10]) == 1.0

    def test_unsorted(self):
        assert min_diff([10, 1, 2]) == 1.0


class TestMaxDiff:
    def test_uniform(self):
        assert max_diff([1, 2, 3, 4]) == 1.0

    def test_uneven(self):
        assert max_diff([1, 2, 10]) == 8.0


class TestNthSmallest:
    def test_first(self):
        assert nth_smallest(np.array([3, 1, 2]), 1) == 1

    def test_second(self):
        assert nth_smallest(np.array([3, 1, 2]), 2) == 2

    def test_third(self):
        assert nth_smallest(np.array([3, 1, 2]), 3) == 3


class TestSecondMin:
    def test_basic(self):
        assert second_min(np.array([5, 1, 3, 2])) == 2

    def test_with_duplicates(self):
        assert second_min(np.array([1, 1, 2, 3])) == 1
