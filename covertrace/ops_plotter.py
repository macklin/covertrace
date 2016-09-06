import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_all(arr, time, ax=None, **kwargs):
    pd.DataFrame(arr.T, index=time).plot(legend=False, ax=ax, **kwargs)


def plot_heatmap(arr, time, ax=None, **kwargs):
    sns.heatmap(arr, ax=ax, **kwargs)


def plot_tsplot(arr, time, ax=None, **kwargs):
    sns.tsplot(arr, time=time, estimator=np.nanmean, ax=ax, **kwargs)


def plot_histogram_pdstats(arr, time, ax, pd_func_name='mean', **keys):
    func = getattr(pd.DataFrame, pd_func_name)
    df_stats = func(pd.DataFrame(arr))
    sns.distplot(df_stats.dropna(), ax=ax, **keys)
