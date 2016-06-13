import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_all(arr, ax=None, **keys):
    pd.DataFrame(arr.T).plot(legend=False, ax=ax, **keys)


def plot_heatmap(arr, ax=None, **keys):
    sns.heatmap(arr, ax=ax, **keys)
