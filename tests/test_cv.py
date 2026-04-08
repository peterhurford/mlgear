import numpy as np
import pandas as pd
import pytest

from mlgear.cv import run_cv_model


def dummy_model(train_X, train_y, test_X=None, test_y=None, test_X2=None,
                params={}, meta=None, verbose=True):
    """Simple model that predicts the training mean."""
    mean_val = np.mean(train_y)
    pred_val = np.full(len(test_y), mean_val) if test_X is not None else None
    pred_test = np.full(test_X2.shape[0], mean_val) if test_X2 is not None else None
    importances = np.ones(train_X.shape[1]) if isinstance(train_X, pd.DataFrame) else None
    return pred_val, pred_test, importances, None


@pytest.fixture
def data():
    np.random.seed(42)
    n = 100
    train = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
    })
    target = np.random.randn(n)
    test = pd.DataFrame({
        'f1': np.random.randn(20),
        'f2': np.random.randn(20),
    })
    return train, target, test


class TestRunCvModel:
    def test_requires_target(self, data):
        train, target, test = data
        with pytest.raises(ValueError, match='Target is needed'):
            run_cv_model(train, model_fn=dummy_model)

    def test_requires_model_fn(self, data):
        train, target, test = data
        with pytest.raises(ValueError, match='model function is needed'):
            run_cv_model(train, target=target)

    def test_basic_cv(self, data):
        train, target, test = data

        def mse(actual, predicted):
            return np.mean((actual - predicted) ** 2)

        results = run_cv_model(
            train, test=test, target=target,
            model_fn=dummy_model, eval_fn=mse,
            label='test', verbose=False
        )
        assert results['label'] == 'test'
        assert len(results['cv']) == 5
        assert results['train'].shape == (100,)
        assert results['test'].shape == (20,)
        assert results['final_cv'] is not None
        assert not results['importance'].empty

    def test_no_test_set(self, data):
        train, target, _ = data

        def mse(actual, predicted):
            return np.mean((actual - predicted) ** 2)

        results = run_cv_model(
            train, target=target,
            model_fn=dummy_model, eval_fn=mse,
            label='test', verbose=False
        )
        assert 'test' not in results
        assert len(results['cv']) == 5

    def test_custom_n_folds(self, data):
        train, target, test = data

        def mse(actual, predicted):
            return np.mean((actual - predicted) ** 2)

        results = run_cv_model(
            train, test=test, target=target,
            model_fn=dummy_model, eval_fn=mse,
            n_folds=3, label='test', verbose=False
        )
        assert len(results['cv']) == 3

    def test_stop_on_fold(self, data):
        train, target, test = data

        def mse(actual, predicted):
            return np.mean((actual - predicted) ** 2)

        results = run_cv_model(
            train, test=test, target=target,
            model_fn=dummy_model, eval_fn=mse,
            stop_on_fold=2, label='test', verbose=False
        )
        assert len(results['cv']) == 2

    def test_no_eval_fn(self, data):
        train, target, test = data
        results = run_cv_model(
            train, test=test, target=target,
            model_fn=dummy_model, eval_fn=None,
            label='test', verbose=False
        )
        assert results['cv'] == []
        assert results['final_cv'] is None

    def test_numpy_array_input(self):
        np.random.seed(42)
        train = np.random.randn(50, 3)
        target = np.random.randn(50)

        def np_model(train_X, train_y, test_X=None, test_y=None, test_X2=None,
                      params={}, meta=None, verbose=True):
            mean_val = np.mean(train_y)
            pred_val = np.full(len(test_y), mean_val) if test_X is not None else None
            pred_test = None
            return pred_val, pred_test, None, None

        results = run_cv_model(
            train, target=target, model_fn=np_model,
            label='np_test', verbose=False
        )
        assert results['train'].shape == (50,)
