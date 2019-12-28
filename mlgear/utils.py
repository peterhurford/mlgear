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


def print_step(step):
    print('[{}] {}'.format(datetime.now(), step))
