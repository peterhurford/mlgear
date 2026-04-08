## v0.6 - development

* Support for `init_score`.
* Add tests
* Lazy-import keras in `metrics.py` and `lr_scheduler.py` so the package is usable without tensorflow
* Fix `runLR` mutating the caller's `params` dict (missing `.copy()`)

## v0.5 - latest release

* `display_columns` now works for dataframes that contain list data.
* Add a new util: `min_max`, to return both the minimum and maximum.
* Updates lightgbm to work on the latest version.
* Supports lightgbm to do ranking.
* Migrated from setuptools/twine to Poetry for package management.

## v0.4

* Add `verbose` parameter to `run_cv_model` and the model functions to silence printing.

## v0.3

* Critical bugfix to make package usable
