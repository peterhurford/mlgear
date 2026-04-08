# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is mlgear?

A Python utility library for ML workflows with pandas/sklearn. Provides cross-validation orchestration, model training wrappers (LightGBM, Keras MLP, Logistic Regression, Ridge), feature encoders, metrics, and a learning rate scheduler.

## Development Commands

```bash
poetry install          # Install dependencies
poetry build            # Build package
poetry publish          # Publish to PyPI
flake8 mlgear/          # Lint (max-line-length: 100)
python -m pytest tests/ -v  # Run tests
```

Tests live in `tests/` and cover aggregators, utils, tracker, encoders, cv, and models. The metrics tests are skipped when tensorflow is not installed (keras backend dependency). Run with `python -m pytest tests/ -v`.

## Architecture

The core workflow is `run_cv_model()` in `cv.py`, which orchestrates K-fold cross-validation by calling a model function for each fold, aggregating predictions, and tracking feature importance.

All model functions in `models.py` (`runLGB`, `runMLP`, `runLR`, `runRidge`) follow the same signature so they're interchangeable with `run_cv_model()`:

```python
def runXXX(train_X, train_y, test_X=None, test_y=None, test_X2=None, params={}, meta=None, verbose=True)
# Returns: (pred_test_y, pred_test_y2, importances, model)
```

- `test_X` / `test_y` = validation fold data (used for early stopping)
- `test_X2` = held-out test set for final predictions
- `params` dict is **mutated** — keys like `num_rounds`, `early_stop`, `group`, `cat_cols`, `feval`, `init_score_col` are popped before passing to the underlying library
- `meta` carries fold metadata (`dev_index`, `val_index`, `fold`, `label`)

Encoders in `encoders.py` are sklearn-compatible (fit/transform): `CategoricalDummyEncoder`, `ValueCountsEncoder`, `BayesTargetEncoder`, `Scaler`.

## Key Details

- LightGBM `runLGB` supports ranking tasks (via `group` param), multi-bag ensembling (`nbag`), and init_score for base margin anchoring with logit/sigmoid post-processing.
- `run_cv_model` defaults to KFold(n_splits=5, shuffle=True, random_state=42) but accepts custom `fold_splits`.
- CRPS metrics in `metrics.py` are hardcoded for 199 bins.
- Poetry is the build system (migrated from setuptools). `requirements.txt` and `setup.cfg` are legacy.
