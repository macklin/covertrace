import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssdistance
import pandas as pd
import numpy as np
from utils.sorting import calc_distance_sqerror, fill_short_series

def sort_hierarchical(raw_arr, dist_func='cosine', NORM=True, FRAME_START=0, FRAME_END=None):
    """cosine, eucledian
    
    Examples:
        >>> arr = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]], np.float32)
        >>> sort_hierarchical(arr)
        array([0, 2, 1, 3])
    """
    
    dataframe = pd.DataFrame(raw_arr.T)
    if NORM is True:
        dataframe = (dataframe - dataframe.mean()) / dataframe.std(ddof = 0)
    arr = dataframe.values
    arr = arr[FRAME_START:FRAME_END, :]
    distance = calc_distance_sqerror(arr, dist_func)
    Y = sch.linkage(distance, method = 'ward')
    Z = sch.dendrogram(Y, orientation = 'right', no_plot = True)
    index = Z['leaves']
    sorted_idx = dataframe.columns[index].values
    return sorted_idx