import numpy as np
import pandas as pd
import pytest

from mlgear.encoders import (
    CategoricalDummyEncoder,
    ValueCountsEncoder,
    BayesTargetEncoder,
    Scaler,
)


@pytest.fixture
def cat_df():
    return pd.DataFrame({
        'color': ['red', 'blue', 'red', 'green'],
        'size': ['S', 'M', 'L', 'M'],
        'value': [1, 2, 3, 4],
    })


class TestCategoricalDummyEncoder:
    def test_fit_auto_detects_object_columns(self, cat_df):
        enc = CategoricalDummyEncoder()
        enc.fit(cat_df)
        assert set(enc.categorical_variables) == {'color', 'size'}

    def test_fit_with_explicit_categories(self, cat_df):
        enc = CategoricalDummyEncoder()
        enc.fit(cat_df, categories=['color'])
        assert enc.categorical_variables == ['color']

    def test_transform_creates_dummies(self, cat_df):
        enc = CategoricalDummyEncoder()
        enc.fit(cat_df)
        result = enc.transform(cat_df.copy())
        assert 'color_red' in result.columns
        assert 'color_blue' in result.columns
        assert 'size_S' in result.columns
        assert 'value' in result.columns

    def test_transform_handles_unseen_categories(self, cat_df):
        enc = CategoricalDummyEncoder()
        enc.fit(cat_df)
        new_df = pd.DataFrame({
            'color': ['red'],
            'size': ['S'],
            'value': [5],
        })
        result = enc.transform(new_df.copy())
        # Missing categories should be filled with 0
        assert 'color_green' in result.columns
        assert result['color_green'].iloc[0] == 0


class TestValueCountsEncoder:
    def test_fit_transform(self, cat_df):
        enc = ValueCountsEncoder()
        enc.fit(cat_df)
        result = enc.transform(cat_df.copy())
        # 'red' appears 2 times, 'blue' 1, 'green' 1
        assert result['color'].iloc[0] == 2  # red
        assert result['color'].iloc[1] == 1  # blue

    def test_with_suffix(self, cat_df):
        enc = ValueCountsEncoder()
        enc.fit(cat_df, categories=['color'])
        result = enc.transform(cat_df.copy(), suffix='_vc')
        assert 'color_vc' in result.columns

    def test_explicit_categories(self, cat_df):
        enc = ValueCountsEncoder()
        enc.fit(cat_df, categories=['color'])
        assert enc.categorical_variables == ['color']

    def test_does_not_mutate_input(self, cat_df):
        enc = ValueCountsEncoder()
        enc.fit(cat_df, categories=['color'])
        original_cols = list(cat_df.columns)
        enc.transform(cat_df, suffix='_vc')
        assert list(cat_df.columns) == original_cols


class TestBayesTargetEncoder:
    def test_fit_transform(self):
        df = pd.DataFrame({
            'cat': ['a', 'a', 'a', 'b', 'b', 'b'],
            'target': [1, 1, 1, 0, 0, 0],
        })
        enc = BayesTargetEncoder()
        enc.fit(df, target='target', weight=0)
        result = enc.transform(df.copy())
        # With weight=0, encoding should be close to group means
        assert result['cat'].iloc[0] == pytest.approx(1.0)
        assert result['cat'].iloc[3] == pytest.approx(0.0)

    def test_smoothing(self):
        df = pd.DataFrame({
            'cat': ['a', 'a', 'b', 'b'],
            'target': [1, 1, 0, 0],
        })
        enc = BayesTargetEncoder()
        enc.fit(df, target='target', weight=10)
        result = enc.transform(df.copy())
        # With high weight, both should be pulled toward global mean (0.5)
        assert result['cat'].iloc[0] < 1.0
        assert result['cat'].iloc[2] > 0.0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            'cat': ['a', 'a', 'b', 'b'],
            'target': [1, 1, 0, 0],
        })
        enc = BayesTargetEncoder()
        enc.fit(df, target='target')
        original_values = list(df['cat'])
        enc.transform(df)
        assert list(df['cat']) == original_values


class TestScaler:
    def test_columns_only(self):
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
            'c': [0, 1, 0, 1, 0],
        })
        scaler = Scaler(columns=['a', 'b'])
        scaler.fit(df)
        result = scaler.transform(df.copy())
        assert list(result.columns) == ['a', 'b', 'c']
        # Unscaled column should remain unchanged
        assert list(result['c']) == [0, 1, 0, 1, 0]

    def test_ordinals_only(self):
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [0, 1, 0],
        })
        scaler = Scaler(ordinals=['a'])
        scaler.fit(df)
        result = scaler.transform(df.copy())
        # Ordinal scaled with MinMaxScaler -> [0, 0.5, 1]
        assert result['a'].iloc[0] == pytest.approx(0.0)
        assert result['a'].iloc[2] == pytest.approx(1.0)

    def test_no_columns_no_ordinals(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        scaler = Scaler()
        scaler.fit(df)
        result = scaler.transform(df.copy())
        assert list(result['a']) == [1, 2, 3]
