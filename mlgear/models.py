import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from mlgear.utils import print_step


def runLGB(train_X, train_y, test_X=None, test_y=None, test_X2=None, params=None, meta=None, verbose=True):
    if params is None:
        params = {}
    if verbose:
        print_step('Prep LGB')

    if params.get('group'):
        group = params.pop('group')
    else:
        group = None

    num_rounds = params.pop('num_rounds')
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

    # Extract init_score if present (for base_margin / odds anchoring)
    init_score_col = params.pop('init_score_col', None)
    train_init_score = None
    test_init_score = None
    if init_score_col and init_score_col in train_X.columns:
        train_init_score = train_X[init_score_col].values
        train_X = train_X.drop(columns=[init_score_col])
        if test_X is not None and init_score_col in test_X.columns:
            test_init_score = test_X[init_score_col].values
            test_X = test_X.drop(columns=[init_score_col])

    if group is None:
        d_train = lgb.Dataset(train_X, label=train_y, init_score=train_init_score)
    else:
        d_train = lgb.Dataset(train_X.drop(group, axis=1),
                              label=train_y,
                              init_score=train_init_score,
                              group=train_X.groupby(group).size().to_numpy())

    if test_X is not None:
        if group is None:
            d_valid = lgb.Dataset(test_X, label=test_y, init_score=test_init_score)
        else:
            d_valid = lgb.Dataset(test_X.drop(group, axis=1),
                                  label=test_y,
                                  init_score=test_init_score,
                                  group=test_X.groupby(group).size().to_numpy())
            test_X = test_X.drop(group, axis=1)
        watchlist = [d_train, d_valid]
    else:
        watchlist = [d_train]

    test_init_score2 = None
    if init_score_col and test_X2 is not None and init_score_col in test_X2.columns:
        test_init_score2 = test_X2[init_score_col].values
        test_X2 = test_X2.drop(columns=[init_score_col])
    if test_X2 is not None and group is not None:
        test_X2 = test_X2.drop(group, axis=1)

    if verbose:
        print_step('Train LGB')

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
            # If init_score was used, predict() returns raw margin (trees only).
            # Add init_score back and apply sigmoid to get probability.
            if test_init_score is not None:
                raw_margin = np.log(pred_test_y.clip(1e-15, 1-1e-15) / (1 - pred_test_y.clip(1e-15, 1-1e-15)))
                pred_test_y = 1 / (1 + np.exp(-(raw_margin + test_init_score)))
            preds_test_y += [pred_test_y]
        if test_X2 is not None:
            if verbose:
                print_step('Predict 2/2')
            pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
            if test_init_score2 is not None:
                raw_margin2 = np.log(pred_test_y2.clip(1e-15, 1-1e-15) / (1 - pred_test_y2.clip(1e-15, 1-1e-15)))
                pred_test_y2 = 1 / (1 + np.exp(-(raw_margin2 + test_init_score2)))
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
    model = lgb.train(lgb_params2, train_d, rounds, valid_sets = [train_d])
    feature_df = pd.DataFrame(sorted(zip(model.feature_importance(), train.columns)),
                               columns=['Value', 'Feature']).sort_values('Value', ascending=False)
    return feature_df


def runMLP(train_X, train_y, test_X=None, test_y=None, test_X2=None, params=None, meta=None, verbose=True):
    if params is None:
        params = {}
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


def runLR(train_X, train_y, test_X=None, test_y=None, test_X2=None, params=None, meta=None, verbose=True):
    if params is None:
        params = {}
    params = params.copy()
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


def runRidge(train_X, train_y, test_X=None, test_y=None, test_X2=None, params=None, meta=None, verbose=True):
    if params is None:
        params = {}
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

