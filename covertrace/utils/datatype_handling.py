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
