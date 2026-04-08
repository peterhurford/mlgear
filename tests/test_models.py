import numpy as np
import pandas as pd
import pytest

from mlgear.models import runLGB, runLR, runRidge


@pytest.fixture
def binary_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
    })
    y = (X['f1'] + X['f2'] > 0).astype(int).values
    return X, y


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
    })
    y = (X['f1'] * 2 + X['f2'] + np.random.randn(n) * 0.1).values
    return X, y


class TestRunLGB:
    def test_basic_train_and_predict(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        pred_val, pred_test, importances, model = runLGB(
            train_X, train_y, test_X, test_y, test_X2=None,
            params=params, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test is None
        assert len(importances) == 2

    def test_with_test_x2(self, binary_data):
        X, y = binary_data
        train_X, test_X, test_X2 = X.iloc[:60], X.iloc[60:80], X.iloc[80:]
        train_y, test_y = y[:60], y[60:80]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        pred_val, pred_test2, importances, model = runLGB(
            train_X, train_y, test_X, test_y, test_X2,
            params=params, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test2.shape == (20,)

    def test_early_stopping(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 100,
            'early_stop': 5,
            'seed': 42,
            'verbose': -1,
        }
        pred_val, _, importances, model = runLGB(
            train_X, train_y, test_X, test_y,
            params=params, verbose=False
        )
        assert model.best_iteration <= 100

    def test_no_validation_set(self, binary_data):
        X, y = binary_data
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        pred_val, pred_test, importances, model = runLGB(
            X, y, params=params, verbose=False
        )
        assert pred_val is None
        assert pred_test is None

    def test_nbag(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'nbag': 3,
            'seed': 42,
            'verbose': -1,
        }
        pred_val, _, _, model = runLGB(
            train_X, train_y, test_X, test_y,
            params=params, verbose=False
        )
        assert pred_val.shape == (20,)


class TestRunLR:
    def test_basic(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        pred_val, pred_test, coefs, model = runLR(
            train_X, train_y, test_X, test_y,
            params={}, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test is None
        assert coefs.shape[1] == 2

    def test_with_scaling(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        pred_val, _, _, model = runLR(
            train_X, train_y, test_X, test_y,
            params={'scale': True}, verbose=False
        )
        assert pred_val.shape == (20,)

    def test_with_test_x2(self, binary_data):
        X, y = binary_data
        train_X, test_X, test_X2 = X.iloc[:60], X.iloc[60:80], X.iloc[80:]
        train_y, test_y = y[:60], y[60:80]
        pred_val, pred_test2, _, _ = runLR(
            train_X, train_y, test_X, test_y, test_X2,
            params={}, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test2.shape == (20,)

    def test_no_validation(self, binary_data):
        X, y = binary_data
        pred_val, pred_test, _, _ = runLR(
            X, y, params={}, verbose=False
        )
        assert pred_val is None
        assert pred_test is None


class TestRunRidge:
    def test_basic(self, regression_data):
        X, y = regression_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        pred_val, pred_test, coefs, model = runRidge(
            train_X, train_y, test_X, test_y,
            params={}, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test is None
        assert len(coefs) == 2

    def test_with_test_x2(self, regression_data):
        X, y = regression_data
        train_X, test_X, test_X2 = X.iloc[:60], X.iloc[60:80], X.iloc[80:]
        train_y, test_y = y[:60], y[60:80]
        pred_val, pred_test2, _, _ = runRidge(
            train_X, train_y, test_X, test_y, test_X2,
            params={}, verbose=False
        )
        assert pred_val.shape == (20,)
        assert pred_test2.shape == (20,)

    def test_no_validation(self, regression_data):
        X, y = regression_data
        pred_val, pred_test, _, _ = runRidge(
            X, y, params={}, verbose=False
        )
        assert pred_val is None
        assert pred_test is None
