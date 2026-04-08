from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np


def show(df: Union[pd.DataFrame, np.ndarray], max_rows: int = 10,
         max_cols: Optional[int] = None, digits: int = 6) -> None:
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', max_cols,
                           'display.float_format', lambda x: '%.{}f'.format(digits) % x):
        print(df)
    if isinstance(df, pd.DataFrame) or isinstance(df, np.ndarray):
        print(df.shape)


def display_column(df: pd.DataFrame, var: str) -> None:
	if df[var].astype(str).nunique() > 9 and (df[var].dtype == int or df[var].dtype == float):
		print('Mean: {} Median: {} SD: {}'.format(df[var].mean(), df[var].median(), df[var].std()))
	else:
		print(df[var].value_counts(normalize=True) * 100)

def display_columns(df: pd.DataFrame) -> None:
	for var in sorted(df.columns):
		print('## {} ##'.format(var))
		display_column(df, var)
		print('-')
		print('-')


def print_step(step: str) -> None:
    print('[{}] {}'.format(datetime.now(), step))


def chunk(l: Sequence[Any], n: int) -> List[list]:
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out


def min_max(dat: Sequence[float]) -> Tuple[float, float]:
    return (min(dat), max(dat))
