# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is mlgear?

A Python utility library for ML workflows with pandas/sklearn. Provides cross-validation orchestration, model training wrappers (LightGBM, Keras MLP, Logistic Regression, Ridge), feature encoders, metrics, and a learning rate scheduler.

## Development Commands

```bash
pip install -e .                # Install in editable mode
poetry install                  # Install dependencies
poetry build                    # Build package
poetry publish                  # Publish to PyPI
flake8 mlgear/                  # Lint (max-line-length: 100)
python -m pytest tests/ -v      # Run tests
```

Tests live in `tests/` and cover aggregators, utils, tracker, encoders, cv, models, and metrics. Run with `python -m pytest tests/ -v`.

## Architecture

The core workflow is `run_cv_model()` in `cv.py`, which orchestrates K-fold cross-validation by calling a model function for each fold, aggregating predictions, and tracking feature importance.

All model functions in `models.py` (`runLGB`, `runMLP`, `runLR`, `runRidge`) follow the same signature so they're interchangeable with `run_cv_model()`:

```python
def runXXX(train_X, train_y, test_X=None, test_y=None, test_X2=None, params=None, meta=None, verbose=True)
# Returns: (pred_test_y, pred_test_y2, importances, model)
```

- `test_X` / `test_y` = validation fold data (used for early stopping)
- `test_X2` = held-out test set for final predictions
- `params` defaults to `None` (replaced with `{}` internally). Keys like `num_rounds`, `early_stop`, `group`, `cat_cols`, `feval`, `init_score_col` are popped before passing to the underlying library. The caller's dict is not mutated — `run_cv_model` and `runLR` copy it internally.
- `meta` carries fold metadata (`dev_index`, `val_index`, `fold`, `label`)

`get_lgb_feature_importance()` is a standalone utility for quick feature importance without full CV.

Encoders in `encoders.py` are sklearn-compatible (fit/transform): `CategoricalDummyEncoder`, `ValueCountsEncoder`, `BayesTargetEncoder`, `Scaler`. All encoder `transform` methods return a copy and do not mutate the input DataFrame.

All public functions have type hints. Public API is exported from `__init__.py` (e.g. `from mlgear import run_cv_model, runLGB`).

## Key Details

- LightGBM `runLGB` supports ranking tasks (via `group` param), multi-bag ensembling (`nbag`), and init_score for base margin anchoring with logit/sigmoid post-processing.
- `run_cv_model` defaults to KFold(n_splits=5, shuffle=True, random_state=42) but accepts custom `fold_splits`.
- CRPS metrics in `metrics.py` use `CRPS_BINS = 199`. `crps_score` validates input shape matches this.
- Keras-dependent code (`k_crps`, `k_crps_`, `runMLP`, `CustomLRScheduler`) uses lazy imports so the rest of the package works without tensorflow.
- Poetry is the build system. Flake8 config lives in `.flake8`. Pytest config lives in `pyproject.toml`.

## Change Workflow

When making changes to this repo:

1. Make each fix/feature as a separate commit.
2. Run `python -m pytest tests/ -v` after each change to verify nothing is broken.
3. Add or update tests for any changed behavior.
4. Log each change in `CHANGES.md` under the current development version.
   - Prefix with category: `Feature -`, `Bugfix -`, `Internal -`
   - Add `**[Breaking change]**` prefix when the change would break previous versions.
   - Sort entries by importance to the user: breaking changes first, then features, bugfixes, internal last.
5. Keep `CLAUDE.md` up to date if the change affects architecture, commands, or key details documented here.
