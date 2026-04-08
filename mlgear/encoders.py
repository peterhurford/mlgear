from typing import Dict, List, Optional, Tuple

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


class CategoricalDummyEncoder(TransformerMixin):
    """Identifies categorical columns by dtype of object and dummy codes them. Optionally a pandas.DataFrame
    can be returned where categories are of pandas.Category dtype and not binarized for better coding strategies
    than dummy coding."""
    def __init__(self, sparse: bool = False) -> None:
        self.categorical_variables: List[str] = []
        self.categories_per_column: Dict = {}
        self.sparse = sparse

    def fit(self, X: pd.DataFrame, categories: Optional[List[str]] = None) -> 'CategoricalDummyEncoder':
        if categories is None:
            self.categorical_variables = list(X.select_dtypes(include=['object']).columns)
        else:
            self.categorical_variables = categories
        for col in self.categorical_variables:
            self.categories_per_column[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(X, sparse=self.sparse, columns=self.categorical_variables)
        expected_columns = sum([['{}_{}'.format(k, v) for v in vs] for k, vs in self.categories_per_column.items()], [])
        for c in expected_columns:
            if c not in X.columns:
                X.loc[:, c] = 0
        return X


class ValueCountsEncoder(TransformerMixin):
    def __init__(self) -> None:
        self.categorical_variables: List[str] = []
        self.categories_per_column: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, categories: Optional[List[str]] = None) -> 'ValueCountsEncoder':
        if categories is None:
            self.categorical_variables = list(X.select_dtypes(include=['object']).columns)
        else:
            self.categorical_variables = categories
        for col in self.categorical_variables:
            self.categories_per_column[col] = dict(X[col].value_counts().items())
        return self

    def transform(self, X: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        X = X.copy()
        for col in self.categorical_variables:
            X.loc[:, '{}{}'.format(col, suffix)] = X[col].map(self.categories_per_column[col])
        return X


# Adapted from https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
class BayesTargetEncoder(TransformerMixin):
    def __init__(self) -> None:
        self.categorical_variables: List[str] = []
        self.categories_per_column: Dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, target: str, weight: float = 10,
            categories: Optional[List[str]] = None) -> 'BayesTargetEncoder':
        if categories is None:
            self.categorical_variables = list(X.select_dtypes(include=['object']).columns)
        else:
            self.categorical_variables = categories
        global_mean = X[target].mean()
        for col in self.categorical_variables:
            agg = X.groupby(col)[target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            smooth = (counts * means + weight * global_mean) / (counts + weight)
            self.categories_per_column[col] = smooth
        return self

    def transform(self, X: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        X = X.copy()
        for col in self.categorical_variables:
            X.loc[:, '{}{}'.format(col, suffix)] = X[col].map(self.categories_per_column[col])
        return X


# https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, ordinals: Optional[List[str]] = None,
                 copy: bool = True, feature_range: Tuple[float, float] = (0, 1)) -> None:
        self.columns = columns
        self.ordinals = ordinals
        self.scaler = PowerTransformer(method='yeo-johnson', standardize=True, copy=True) if columns else None
        self.ordinal_scaler = MinMaxScaler(copy=copy, feature_range=feature_range) if ordinals else None

    def fit(self, X: pd.DataFrame, y=None) -> 'Scaler':
        if self.columns:
            self.scaler.fit(X[self.columns], y)
        if self.ordinals:
            self.ordinal_scaler.fit(X[self.ordinals], y)
        return self

    def transform(self, X: pd.DataFrame, y=None, copy=None) -> pd.DataFrame:
        init_col_order = X.columns
        if self.columns:
            X_scaled = X[self.columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X_scaled), columns=self.columns, index=X.index)
        if self.ordinals:
            X_ordinals = X[self.ordinals]
            X_ordinals = pd.DataFrame(self.ordinal_scaler.transform(X_ordinals), columns=self.ordinals, index=X.index)
        if self.columns and self.ordinals:
            X_not_scaled = X[list(set(init_col_order) - set(self.columns) - set(self.ordinals))]
            return pd.concat([X_not_scaled, X_scaled, X_ordinals], axis=1)[init_col_order]
        elif self.columns:
            X_not_scaled = X[list(set(init_col_order) - set(self.columns))]
            return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
        elif self.ordinals:
            X_not_scaled = X[list(set(init_col_order) - set(self.ordinals))]
            return pd.concat([X_not_scaled, X_ordinals], axis=1)[init_col_order]
        else:
            return X
