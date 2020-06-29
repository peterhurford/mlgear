## MLGear

Some utility functions to make ML with Python / Pandas / sklearn even easier

#### Example Usage

```Python
from mlgear.models import runLGB
from mlgear.metrics import rmse

lgb_params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 15,
              'learning_rate': 0.01,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.9,
              'verbosity': -1,
              'seed': 1,
              'lambda_l1': 1,
              'lambda_l2': 1,
              'early_stop': 20,
              'verbose_eval': 10,
              'num_rounds': 500,
              'num_threads': 3}

results = run_cv_model(train, test, target, runLGB, lgb_params, rmse)
```
