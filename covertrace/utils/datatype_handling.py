import pandas as pd
from itertools import product
import numpy as np
from os.path import basename, join, dirname


def pd_array_convert(path):
    df = pd.read_csv(path, index_col=['object', 'ch', 'prop', 'frame'])
    objects, channels, props = [list(i) for i in df.index.levels[:3]]
    labels = [i for i in product(objects, channels, props)]
    storage = []
    for i in labels:
        storage.append(np.float32(df.ix[i]).T)
    arr = np.rollaxis(np.dstack(storage), 2)

    dic_save = {}
    dic_save['data'] = arr
    dic_save['labels'] = labels

    # FILE NAME
    file_name = basename(path).split('.')[0]
    np.savez_compressed(join(dirname(path), file_name), **dic_save)


def save_output(arr, labels, path):
    dic_save = {'data': arr, 'labels': labels}
    np.savez_compressed(path, **dic_save)


def sort_labels_and_arr(labels, arr=[]):
    '''
    >>> labels = [['a', 'B', '1'], ['a', 'A', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> sort_labels(labels)
    [['a', 'A', '1'], ['a', 'B', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> labels = [['a', 'B', '1'], ['prop'], ['aprop'], ['b', 'B', '2']]
    >>> sort_labels(labels)
    [['aprop'], ['prop'], ['a', 'B', '1'], ['b', 'B', '2']]
    '''
    single_labels, single_idx = sort_multi_labels([a for a in labels if len(a) == 1])
    multi_labels, multi_idx = sort_multi_labels([a for a in labels if len(a) == 3])
    # sort_idx = single_idx + [i + len(single_idx) for i in multi_idx]
    sort_idx = []
    for i in labels:
        if i in multi_labels:
            sort_idx.append(multi_labels.index(i) + len(single_labels))
        elif i in single_labels:
            sort_idx.append(single_labels.index(i))
    labels = [labels[i] for i in sort_idx]
    if not len(arr):
        return labels
    if len(arr):
        arr = arr[sort_idx, :, :]
        return labels, arr


def sort_multi_labels(labels):
    '''
    >>> sort_multi_labels([['b'], ['c'], ['a']])
    ([['a'], ['b'], ['c']], [2, 0, 1])
    >>> labels = [['a', 'B', '1'], ['a', 'A', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> sort_multi_labels(labels)
    ([['a', 'A', '1'], ['a', 'B', '1'], ['b', 'A', '3'], ['b', 'B', '2']], [1, 0, 2, 3])
    '''
    sort_idx = []
    if labels:
        if len(labels[0]) == 1:
            sort_func = lambda x: (x[1][0])
        elif len(labels[0]) == 2:
            sort_func = lambda x: (x[1][0], x[1][1])
        elif len(labels[0]) == 3:
            sort_func = lambda x: (x[1][0], x[1][1], x[1][2])
        sort_idx = [i[0] for i in sorted(enumerate(labels), key=sort_func)]
        labels = [labels[i] for i in sort_idx]
    return labels, sort_idx
