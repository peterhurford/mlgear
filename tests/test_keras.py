import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
keras = pytest.importorskip('keras')

from mlgear.lr_scheduler import CustomLRScheduler  # noqa: E402
from mlgear.metrics import CRPS_BINS, crps_score_, k_crps, k_crps_  # noqa: E402
from mlgear.models import runMLP  # noqa: E402


@pytest.fixture
def mlp_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((64, 3)).astype('float32')
    y = (X.sum(axis=1) > 0).astype('float32')
    return X, y


def _make_model(input_size, lr=0.01):
    model = keras.Sequential([
        keras.Input(shape=(input_size,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
    return model


def _lr(model):
    return float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))


def _crps_inputs():
    rng = np.random.default_rng(0)
    actual = np.zeros((5, CRPS_BINS), dtype='float32')
    actual[np.arange(5), rng.integers(0, CRPS_BINS, 5)] = 1
    predicted = rng.dirichlet(np.ones(CRPS_BINS), 5).astype('float32')
    return actual, predicted


class TestKCrps:
    def test_matches_numpy_crps(self):
        actual, predicted = _crps_inputs()
        result = float(k_crps(tf.constant(actual), tf.constant(predicted)))
        assert result == pytest.approx(crps_score_(actual, predicted), rel=1e-5)

    def test_unnormalized_sum(self):
        actual, predicted = _crps_inputs()
        result = float(k_crps_(tf.constant(actual), tf.constant(predicted)))
        expected = crps_score_(actual, predicted) * CRPS_BINS * actual.shape[0]
        assert result == pytest.approx(expected, rel=1e-5)


class TestCustomLRScheduler:
    def test_warmup_raises_lr(self, mlp_data):
        X, y = mlp_data
        model = _make_model(3, lr=0.001)
        sched = CustomLRScheduler(warmup_epochs=3, max_lr=0.01, min_lr=0.0001)
        model.fit(X, y, epochs=2, verbose=0, validation_data=(X, y), callbacks=[sched])
        assert _lr(model) > 0.001

    def test_patience_lowers_lr(self, mlp_data):
        X, y = mlp_data
        model = _make_model(3, lr=0.01)
        sched = CustomLRScheduler(warmup_epochs=0, max_lr=0.01, min_lr=0.0001,
                                  patience=2, factor=0.5)
        sched.set_model(model)
        sched.on_epoch_end(0, {'val_loss': 1.0})
        sched.on_epoch_end(1, {'val_loss': 2.0})
        assert _lr(model) == pytest.approx(0.005)


class TestRunMLP:
    def test_basic_train_and_predict(self, mlp_data):
        X, y = mlp_data
        params = {
            'model': _make_model,
            'input_size': 3,
            'early_stopper': lambda: keras.callbacks.EarlyStopping(patience=2),
            'metric': lambda model, callbacks, data: keras.callbacks.Callback(),
            'lr_scheduler': lambda: [],
            'epochs': 3,
            'batch_size': 16,
        }
        pred, pred2, importances, model = runMLP(X[:48], y[:48], X[48:], y[48:], X[48:],
                                                 params=params, verbose=False)
        assert pred.shape == (16, 1)
        assert pred2.shape == (16, 1)
        assert importances is None
        assert np.all((pred >= 0) & (pred <= 1))
