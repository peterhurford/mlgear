import numpy as np

from math import sqrt
from sklearn.metrics import mean_squared_error


CRPS_BINS = 199


def crps_score(actual, predicted):
    if actual.shape[1] != CRPS_BINS:
        raise ValueError(
            'crps_score expects {} bins, got {} columns'.format(CRPS_BINS, actual.shape[1])
        )
    return ((predicted - actual) ** 2).sum(axis=1).sum(axis=0) / (CRPS_BINS * actual.shape[0])

def crps_score_(actual, predicted):
    actual = np.clip(np.cumsum(actual, axis=1), 0, 1)
    predicted = np.clip(np.cumsum(predicted, axis=1), 0, 1)
    return crps_score(actual, predicted)

def k_crps(y_true, y_pred):
    from keras import backend as K
    y_true = K.clip(K.cumsum(y_true, axis=1), 0, 1)
    y_pred = K.clip(K.cumsum(y_pred, axis=1), 0, 1)
    return K.sum(K.sum(K.square(y_true - y_pred), axis=1), axis=0) / K.cast(CRPS_BINS * K.shape(y_true)[0], 'float32')

def k_crps_(y_true, y_pred):
    from keras import backend as K
    y_true = K.clip(K.cumsum(y_true, axis=1), 0, 1)
    y_pred = K.clip(K.cumsum(y_pred, axis=1), 0, 1)
    return K.sum(K.sum(K.square(y_true - y_pred), axis=1), axis=0)

def crps_lgb(actual, predicted):
    actual_ = np.zeros((actual.shape[0], CRPS_BINS))
    for idx, target in enumerate(list(actual)):
        actual_[idx][int(target)] = 1
    return crps_score_(actual_, predicted)

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
