## v0.6 - development

* **[Breaking change]** Internal - Bump minimum Python version from 3.7 (EOL) to 3.11 (required by keras 3.15)
* **[Breaking change]** Internal - Require `keras>=3.15.0`, `numpy>=1.22` and `scikit-learn>=1.5.0` to clear security advisories; relock (Pygments 2.21.0, zipp dropped)
* **[Breaking change]** Feature - Add shape validation to `crps_score` and extract `CRPS_BINS` constant for the hardcoded 199
* Feature - Support for `init_score`.
* Feature - Lambdarank predictions now correctly reconstitute `init_score` (raw score + init_score → sigmoid) instead of using the binary logit round-trip.
* Feature - Support for per-row `sample_weight` in `runLGB` via the `sample_weight_col` param (mirrors `init_score_col`).
* Feature - Add public API exports to `__init__.py` (e.g. `from mlgear import run_cv_model, runLGB`)
* Bugfix - Port `k_crps`, `k_crps_` and `CustomLRScheduler` to keras 3 (`keras.ops`, `optimizer.learning_rate`); they raised `AttributeError` on any keras 3
* Bugfix - Fix `runLR` mutating the caller's `params` dict (missing `.copy()`)
* Bugfix - Fix `ValueCountsEncoder` and `BayesTargetEncoder` mutating the input DataFrame during `transform`
* Bugfix - Replace mutable default `params={}` with `params=None` in all model functions and `run_cv_model`
* Bugfix - Fix `Scaler` crashing with `AttributeError` when only `columns` or only `ordinals` is set
* Internal - Lazy-import keras in `metrics.py` and `lr_scheduler.py` so the package is usable without tensorflow
* Internal - Add type hints to all public functions
* Internal - Add tests
* Internal - Add optional `keras` dev group (`poetry install --with keras`) with tensorflow, and keras tests that skip without it
* Internal - Add `pytest` to dev dependencies
* Internal - Remove legacy `requirements.txt` and `setup.cfg` (flake8 config moved to `.flake8`)

## v0.5 - latest release

* **[Breaking change]** Internal - Migrated from setuptools/twine to Poetry for package management.
* Feature - Supports lightgbm to do ranking.
* Feature - Add a new util: `min_max`, to return both the minimum and maximum.
* Bugfix - `display_columns` now works for dataframes that contain list data.
* Internal - Updates lightgbm to work on the latest version.

## v0.4

* Feature - Add `verbose` parameter to `run_cv_model` and the model functions to silence printing.

## v0.3

* Bugfix - Critical bugfix to make package usable
