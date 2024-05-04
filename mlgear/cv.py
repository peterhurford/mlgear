import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from mlgear.utils import print_step


def run_cv_model(train, test=None, target=None, model_fn=None, params={}, eval_fn=None, label='model', n_folds=5, fold_splits=None, classes=1, stop_on_fold=None, train_on_full=False, verbose=True):
    if target is None:
        raise ValueError('Target is needed.')
    if model_fn is None:
        raise ValueError('model function is needed.')
    if not fold_splits:
        kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
        fold_splits = kf.split(train)
    cv_scores = []
    models = {}
    if classes > 1 and test is not None:
       pred_full_test = np.zeros((test.shape[0], classes))
    else:
       pred_full_test = 0
    if classes > 1:
       pred_train = np.zeros((train.shape[0], classes))
    else:
       pred_train = np.zeros(train.shape[0])
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        if verbose:
            print_step('Started ' + label + ' fold ' + str(i) + '/' + str(n_folds))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        meta = {'dev_index': dev_index,
                'val_index': val_index,
                'fold': i,
                'label': label}
        pred_val_y, pred_test_y, importances, model = model_fn(dev_X, dev_y, val_X, val_y, test, params2, meta, verbose=verbose)
        if test is not None:
            pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            if verbose:
                print_step(label + ' cv score {}: {}'.format(i, cv_score))
        models[i] = model
        if importances is not None and isinstance(train, pd.DataFrame):
            fold_importance_df = pd.DataFrame()
            if params.get('group') is None:
                fold_importance_df['feature'] = train.columns.values
            else:
                fold_importance_df['feature'] = [c for c in train.columns.values if c != params['group']]
            fold_importance_df['importance'] = importances
            fold_importance_df['fold'] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if stop_on_fold and stop_on_fold == i:
            results = {'label': label,
                       'train': pred_train,
                       'cv': cv_scores,
                       'importance': feature_importance_df,
                       'model': models}
            if test is not None:
                results['test'] = pred_full_test
            return results
        i += 1

    if train_on_full:
        if verbose:
            print_step('## Training on full ##')
        params2 = params.copy()
        _, pred_full_test, importances, model = model_fn(train, target, None, None, test, params2, verbose=verbose)
        models['full'] = model
    elif test is not None:
        pred_full_test = pred_full_test / n_folds
   
    final_cv = eval_fn(target, pred_train) if eval_fn else None

    if verbose:
        print_step('{} cv scores : {}'.format(label, cv_scores))
        print_step('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
        print_step('{} cv total score : {}'.format(label, final_cv))
        print_step('{} cv std score : {}'.format(label, np.std(cv_scores)))

    results = {'label': label,
               'train': pred_train,
               'cv': cv_scores,
               'final_cv': final_cv,
               'importance': feature_importance_df,
               'model': models}
    if test is not None:
        results['test'] = pred_full_test
    return results 
