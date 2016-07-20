import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_all(arr, ax=None, **kwargs):
    pd.DataFrame(arr.T).plot(legend=False, ax=ax, **kwargs)

def plot_heatmap(arr, ax=None, **kwargs):
    sns.heatmap(arr, ax=ax, **kwargs)


def plot_tsplot(arr, ax=None, **kwargs):
    sns.tsplot(arr, estimator=np.nanmean, ax=ax, **kwargs)


def plot_histogram_pdstats(arr, ax, pd_func_name='mean', **keys):
    func = getattr(pd.DataFrame, pd_func_name)
    df_stats = func(pd.DataFrame(arr))
    sns.distplot(df_stats.dropna(), ax=ax, **keys)
