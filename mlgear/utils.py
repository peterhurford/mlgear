from datetime import datetime

import pandas as pd
import numpy as np


def show(df, max_rows=10, max_cols=None, digits=6):
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', max_cols,
                           'display.float_format', lambda x: '%.{}f'.format(digits) % x):
        print(df)
    if isinstance(df, pd.DataFrame) or isinstance(df, np.ndarray):
        print(df.shape)


def display_column(df, var):
	if df[var].astype(str).nunique() > 9 and (df[var].dtype == int or df[var].dtype == float):
		print('Mean: {} Median: {} SD: {}'.format(df[var].mean(), df[var].median(), df[var].std()))
	else:
		print(df[var].value_counts(normalize=True) * 100)

def display_columns(df):
	for var in sorted(df.columns):
		print('## {} ##'.format(var))
		display_column(df, var)
		print('-')
		print('-')


def print_step(step):
    print('[{}] {}'.format(datetime.now(), step))


def chunk(l, n):
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out


def min_max(dat):
    return (min(dat), max(dat))
