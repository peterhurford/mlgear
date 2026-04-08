import numpy as np
import pandas as pd

from mlgear.utils import chunk, min_max, print_step, show, display_column


class TestChunk:
    def test_even_split(self):
        assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_larger_chunk_than_list(self):
        assert chunk([1, 2], 5) == [[1, 2]]

    def test_single_element_chunks(self):
        assert chunk([1, 2, 3], 1) == [[1], [2], [3]]

    def test_empty_list(self):
        assert chunk([], 3) == []


class TestMinMax:
    def test_basic(self):
        assert min_max([3, 1, 4, 1, 5]) == (1, 5)

    def test_single(self):
        assert min_max([7]) == (7, 7)

    def test_negative(self):
        assert min_max([-5, -1, -3]) == (-5, -1)


class TestPrintStep:
    def test_prints_message(self, capsys):
        print_step('hello')
        captured = capsys.readouterr()
        assert 'hello' in captured.out
        assert '[' in captured.out  # contains timestamp


class TestShow:
    def test_dataframe(self, capsys):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        show(df)
        captured = capsys.readouterr()
        assert '(2, 2)' in captured.out

    def test_numpy_array(self, capsys):
        arr = np.array([[1, 2], [3, 4]])
        show(arr)
        captured = capsys.readouterr()
        assert '(2, 2)' in captured.out


class TestDisplayColumn:
    def test_numeric_column(self, capsys):
        df = pd.DataFrame({'x': list(range(20))})
        display_column(df, 'x')
        captured = capsys.readouterr()
        assert 'Mean' in captured.out

    def test_categorical_column(self, capsys):
        df = pd.DataFrame({'x': ['a', 'b', 'a']})
        display_column(df, 'x')
        captured = capsys.readouterr()
        # value_counts output
        assert 'a' in captured.out or '66' in captured.out
