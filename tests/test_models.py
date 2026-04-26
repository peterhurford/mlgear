from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from mlgear.models import runLGB, runLR, runRidge, get_lgb_feature_importance, _sigmoid, _logit


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

    def test_sample_weight_col(self, binary_data):
        X, y = binary_data
        # Build a weight column that heavily upweights one class so the
        # weighted model produces materially different predictions than
        # an unweighted one trained on otherwise-identical data.
        weights = np.where(y == 1, 10.0, 0.1)
        X_with_w = X.copy()
        X_with_w['_sample_weight'] = weights

        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_X_w, test_X_w = X_with_w.iloc[:80], X_with_w.iloc[80:]
        train_y, test_y = y[:80], y[80:]

        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 30,
            'seed': 42,
            'verbose': -1,
        }
        weighted_params = dict(base_params, sample_weight_col='_sample_weight')

        pred_unweighted, _, imp_unweighted, model_unweighted = runLGB(
            train_X, train_y, test_X, test_y,
            params=dict(base_params), verbose=False
        )
        pred_weighted, _, imp_weighted, model_weighted = runLGB(
            train_X_w, train_y, test_X_w, test_y,
            params=weighted_params, verbose=False
        )

        # Weight column must not be a feature in the trained model.
        assert '_sample_weight' not in model_weighted.feature_name()
        assert set(model_weighted.feature_name()) == {'f1', 'f2'}
        # Importance vector length matches the actual feature count (no weight col).
        assert len(imp_weighted) == 2
        # Heavy positive-class weighting should pull predictions upward
        # vs the unweighted baseline.
        assert pred_weighted.mean() > pred_unweighted.mean()
        # And the predictions should not be identical.
        assert not np.allclose(pred_weighted, pred_unweighted)

    def test_sample_weight_col_with_init_score(self, binary_data):
        # Both init_score_col and sample_weight_col set simultaneously.
        X, y = binary_data
        X2 = X.copy()
        X2['init_score'] = 0.0  # neutral base margin
        X2['_sample_weight'] = np.where(y == 1, 5.0, 1.0)

        train_X, test_X, test_X2 = X2.iloc[:60], X2.iloc[60:80], X2.iloc[80:]
        train_y, test_y = y[:60], y[60:80]

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 20,
            'seed': 42,
            'verbose': -1,
            'init_score_col': 'init_score',
            'sample_weight_col': '_sample_weight',
        }
        pred_val, pred_test2, importances, model = runLGB(
            train_X, train_y, test_X, test_y, test_X2,
            params=params, verbose=False
        )
        # Neither helper column should leak into the model.
        assert set(model.feature_name()) == {'f1', 'f2'}
        assert len(importances) == 2
        assert pred_val.shape == (20,)
        assert pred_test2.shape == (20,)

    def test_sample_weight_col_does_not_mutate_input(self, binary_data):
        X, y = binary_data
        X_with_w = X.copy()
        X_with_w['_sample_weight'] = 1.0
        train_X, test_X = X_with_w.iloc[:80], X_with_w.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 5,
            'seed': 42,
            'verbose': -1,
            'sample_weight_col': '_sample_weight',
        }
        runLGB(train_X, train_y, test_X, test_y, params=params, verbose=False)
        # Caller's frames must still contain the weight column.
        assert '_sample_weight' in train_X.columns
        assert '_sample_weight' in test_X.columns


def _mock_lgb_model(predict_values):
    """Create a mock LGB model that returns fixed predict values."""
    model = MagicMock()
    model.best_iteration = 10
    model.feature_importance.return_value = np.array([1, 1])
    model.predict.return_value = predict_values
    return model


class TestLambdarankInitScore:
    """Test init_score reconstitution for both binary and lambdarank objectives."""

    def test_binary_init_score_reconstitution(self):
        """Binary predict() returns probabilities; verify logit→add→sigmoid round-trip."""
        raw_probs = np.array([0.3, 0.7, 0.5])
        init_scores = np.array([0.5, -0.5, 1.0])
        model = _mock_lgb_model(raw_probs)

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'is': init_scores})
        y = np.array([0, 1, 1])
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_rounds': 10, 'seed': 42, 'verbose': -1,
            'init_score_col': 'is',
        }
        with patch('mlgear.models.lgb.train', return_value=model):
            with patch('mlgear.models.lgb.Dataset'):
                pred, _, _, _ = runLGB(X, y, X.copy(), y, params=params, verbose=False)

        expected = _sigmoid(_logit(raw_probs) + init_scores)
        np.testing.assert_allclose(pred, expected)

    def test_lambdarank_init_score_reconstitution(self):
        """Lambdarank predict() returns raw scores; verify score+init→sigmoid."""
        raw_scores = np.array([-1.0, 0.0, 2.0])
        init_scores = np.array([0.5, -0.5, 1.0])
        model = _mock_lgb_model(raw_scores)

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'is': init_scores})
        y = np.array([0, 1, 1])
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg',
            'num_rounds': 10, 'seed': 42, 'verbose': -1,
            'init_score_col': 'is',
        }
        with patch('mlgear.models.lgb.train', return_value=model):
            with patch('mlgear.models.lgb.Dataset'):
                pred, _, _, _ = runLGB(X, y, X.copy(), y, params=params, verbose=False)

        expected = _sigmoid(raw_scores + init_scores)
        np.testing.assert_allclose(pred, expected)

    def test_lambdarank_without_init_score_returns_raw(self):
        """Lambdarank without init_score should return raw scores, not sigmoid."""
        raw_scores = np.array([-1.0, 0.0, 2.0])
        model = _mock_lgb_model(raw_scores)

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        y = np.array([0, 1, 1])
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg',
            'num_rounds': 10, 'seed': 42, 'verbose': -1,
        }
        with patch('mlgear.models.lgb.train', return_value=model):
            with patch('mlgear.models.lgb.Dataset'):
                pred, _, _, _ = runLGB(X, y, X.copy(), y, params=params, verbose=False)

        # Raw scores returned unchanged — no sigmoid applied
        np.testing.assert_allclose(pred, raw_scores)

    def test_lambdarank_init_score_test_x2(self):
        """Verify init_score reconstitution works for test_X2 as well."""
        val_scores = np.array([0.1, 0.2, 0.3])
        x2_scores = np.array([-0.5, 1.5])
        x2_init = np.array([0.2, -0.3])

        model = MagicMock()
        model.best_iteration = 10
        model.feature_importance.return_value = np.array([1, 1])
        model.predict.side_effect = [val_scores, x2_scores]

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'is': [0.0, 0.0, 0.0]})
        X2 = pd.DataFrame({'f1': [7, 8], 'f2': [9, 10], 'is': x2_init})
        y = np.array([0, 1, 1])
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg',
            'num_rounds': 10, 'seed': 42, 'verbose': -1,
            'init_score_col': 'is',
        }
        with patch('mlgear.models.lgb.train', return_value=model):
            with patch('mlgear.models.lgb.Dataset'):
                _, pred2, _, _ = runLGB(X, y, X.copy(), y, X2.copy(),
                                        params=params, verbose=False)

        expected = _sigmoid(x2_scores + x2_init)
        np.testing.assert_allclose(pred2, expected)


class TestSigmoidLogit:
    def test_sigmoid_known_values(self):
        np.testing.assert_allclose(_sigmoid(np.array([0.0])), [0.5])
        assert _sigmoid(np.array([100.0]))[0] > 0.999

    def test_logit_inverts_sigmoid(self):
        p = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(_sigmoid(_logit(p)), p)

    def test_logit_clips_extremes(self):
        # Should not produce inf/-inf for 0.0 and 1.0
        result = _logit(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(result))


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

    def test_does_not_mutate_params(self, binary_data):
        X, y = binary_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        params = {'scale': True}
        runLR(train_X, train_y, test_X, test_y, params=params, verbose=False)
        assert 'scale' in params
        assert 'random_state' not in params


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

    def test_params_none_default(self, regression_data):
        X, y = regression_data
        train_X, test_X = X.iloc[:80], X.iloc[80:]
        train_y, test_y = y[:80], y[80:]
        pred_val, _, _, _ = runRidge(
            train_X, train_y, test_X, test_y, verbose=False
        )
        assert pred_val.shape == (20,)


class TestGetLgbFeatureImportance:
    def test_returns_dataframe(self, binary_data):
        X, y = binary_data
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        result = get_lgb_feature_importance(X, y, params)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Value', 'Feature']
        assert len(result) == 2
        assert set(result['Feature']) == {'f1', 'f2'}

    def test_sorted_descending(self, binary_data):
        X, y = binary_data
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        result = get_lgb_feature_importance(X, y, params)
        assert result['Value'].iloc[0] >= result['Value'].iloc[1]

    def test_does_not_mutate_params(self, binary_data):
        X, y = binary_data
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_rounds': 10,
            'seed': 42,
            'verbose': -1,
        }
        original_keys = set(params.keys())
        get_lgb_feature_importance(X, y, params)
        assert set(params.keys()) == original_keys
