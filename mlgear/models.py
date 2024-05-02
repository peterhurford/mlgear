import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from mlgear.utils import print_step


def runLGB(train_X, train_y, test_X=None, test_y=None, test_X2=None, params={}, meta=None, verbose=True):
    if verbose:
        print_step('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    if test_X is not None:
        d_valid = lgb.Dataset(test_X, label=test_y)
        watchlist = [d_train, d_valid]
    else:
        watchlist = [d_train]
    if verbose:
        print_step('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    if params.get('nbag'):
        nbag = params.pop('nbag')
    else:
        nbag = 1
    if params.get('cat_cols'):
        cat_cols = params.pop('cat_cols')
    else:
        cat_cols = []
    if params.get('feval'):
        feval = params.pop('feval')
    else:
        feval = None

    preds_test_y = []
    preds_test_y2 = []
    for b in range(nbag):
        params['seed'] += b
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          callbacks=[lgb.early_stopping(stopping_rounds=early_stop)] if early_stop else [],
                          feval=feval)
        if test_X is not None:
            if verbose:
                print_step('Predict 1/2')
            pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
            preds_test_y += [pred_test_y]
        if test_X2 is not None:
            if verbose:
                print_step('Predict 2/2')
            pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
            preds_test_y2 += [pred_test_y2]

    if test_X is not None:
        pred_test_y = np.mean(preds_test_y, axis=0)
    else:
        pred_test_y = None
    if test_X2 is not None:
        pred_test_y2 = np.mean(preds_test_y2, axis=0)
    else:
        pred_test_y2 = None
    return pred_test_y, pred_test_y2, model.feature_importance(), model


def get_lgb_feature_importance(train, target, params):
    train_d = lgb.Dataset(train, label=target)
    lgb_params2 = params.copy()
    rounds = lgb_params2.pop('num_rounds', 400)
    verbose_eval = lgb_params2.pop('verbose_eval', 100)
    model = lgb.train(lgb_params2, train_d, rounds, valid_sets = [train_d], verbose_eval=verbose_eval)
    feature_df = pd.DataFrame(sorted(zip(model.feature_importance(), train.columns)),
                               columns=['Value', 'Feature']).sort_values('Value', ascending=False)
    return feature_df


def runMLP(train_X, train_y, test_X=None, test_y=None, test_X2=None, params={}, meta=None, verbose=True):
    if verbose:
        print_step('Define Model')
    model = params['model'](params['input_size'])
    es = params['early_stopper']()
    es.set_model(model)
    metric = params['metric']
    metric = metric(model, [es], [(train_X, train_y), (test_X, test_y)])
    if verbose:
        print_step('Fit MLP')
    model.fit(train_X, train_y,
              verbose=params.get('model_verbose', 0),
              callbacks=[metric] + params['lr_scheduler'](),
              epochs=params.get('epochs', 1000),
              validation_data=(test_X, test_y),
              batch_size=params.get('batch_size', 128))
    if test_X is not None:
        if verbose:
            print_step('MLP Predict 1/2')
        pred_test_y = model.predict(test_X)
    else:
        pred_test_y = None
    if test_X2 is not None:
        if verbose:
            print_step('MLP Predict 2/2')
        pred_test_y2 = model.predict(test_X2)
    else:
        pred_test_y2 = None
    # TODO: Return history object
    return pred_test_y, pred_test_y2, None, model


def runLR(train_X, train_y, test_X=None, test_y=None, test_X2=None, params={}, meta=None, verbose=True):
    params['random_state'] = 42
    if params.get('scale'):
        if verbose:
            print_step('Scale')
        params.pop('scale')
        scaler = StandardScaler()
        scaler.fit(train_X.values)
        train_X = scaler.transform(train_X.values)
        if test_X is not None:
            test_X = scaler.transform(test_X.values)
        if test_X2 is not None:
            test_X2 = scaler.transform(test_X2.values)

    if verbose:
        print_step('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    if test_X is not None:
        if verbose:
            print_step('Predict 1/2')
        pred_test_y = model.predict_proba(test_X)[:, 1]
    else:
        pred_test_y = None
    if test_X2 is not None:
        if verbose:
            print_step('Predict 2/2')
        pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    else:
        pred_test_y2 = None
    return pred_test_y, pred_test_y2, model.coef_, model


def runRidge(train_X, train_y, test_X=None, test_y=None, test_X2=None, params={}, meta=None, verbose=True):
    model = Ridge(**params)
    if verbose:
        print_step('Fit Ridge')
    model.fit(train_X, train_y)
    if test_X is not None:
        if verbose:
            print_step('Ridge Predict 1/2')
        pred_test_y = model.predict(test_X)
    else:
        pred_test_y = None
    if test_X2 is not None:
        if verbose:
            print_step('Ridge Predict 2/2')
        pred_test_y2 = model.predict(test_X2)
    else:
        pred_test_y2 = None
    return pred_test_y, pred_test_y2, model.coef_, model

