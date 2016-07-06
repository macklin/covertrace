'''Input arr is 2D. Make sure output is a slice of original array.
'''
from __future__ import division
import numpy as np
from functools import partial
from utils.array_handling import extend_true, skip_outside_frame_start_to_end
import pandas as pd



def filter_frames_by_range(arr, LOWER=-10000, UPPER=10000, FRAME_START=0, FRAME_END=None):
    """Replace values with NaN if it's not in a range specified by LOWER and UPPER.
    FRAME_START and FRAME_END will determine which frames to look at.

    Examples:
        >>> arr = np.array([[0, 0, 0], [0, 100, 0]], np.float32)
        >>> filter_frames_by_range(arr, UPPER=1)
        array([[False, False, False],
               [False,  True, False]], dtype=bool)
    """
    arr_bool = (arr < UPPER) * (arr > LOWER)
    arr_bool[:, :FRAME_START] = True
    if isinstance(FRAME_END, int):
        arr_bool[:, FRAME_END:] = True
    return -arr_bool


def filter_frames_by_diff(arr, pd_func_name='diff', PERIOD=1, THRES=0.1, FRAME_START=0,
                          FRAME_END=None, LEFT=0, RIGHT=0):
    """Outlier detection by diff or pct_change.
    Replace values with NaN based on diff or pct_change. (may choose eitherfor pd_func_name.)
    FRAME_START and FRAME_END will determine which frames to filter.
    LEFT and RIGHT will extend NaN from the outliers found.
    e.g. For cell death, RIGHT can be a large number so that you can filter out all values after
    a sharp spike.
    Use slider_filter_frames_by_diff to play with parameters.

    Examples:

        >>> arr = np.array([[0, 10, 0], [0, 100, 0]], np.float32)
        >>> filter_frames_by_diff(arr, THRES=15)
        array([[False, False, False],
               [False,  True, False]], dtype=bool)
    """
    above_thres = pd.DataFrame(arr).diff(axis=1).values > THRES
    fn = partial(extend_true, LEFT=LEFT, RIGHT=RIGHT)
    above_thres = np.apply_along_axis(fn, axis=1, arr=above_thres)

    above_thres = skip_outside_frame_start_to_end(above_thres, FRAME_START, FRAME_END)
    return above_thres


def filter_from_last_frames(arr, FRAME_START=0, FRAME_END=None, LEFT=0):
    """Find NaNs and propagate NaNs to previous frames.
    LEFT is how many frames you want to go back.

    Examples:
        >>> arr = np.array([[0, np.nan, np.nan], [0, 0, np.nan]], np.float32)
        >>> filter_from_last_frames(arr, LEFT=1)
        array([[ True,  True, False],
               [False,  True,  True]], dtype=bool)
    """
    df = pd.DataFrame(arr)
    nan_appeared = df.isnull().diff(axis=1).values
    nan_appeared = skip_outside_frame_start_to_end(nan_appeared, FRAME_START, FRAME_END) == 1
    fn = partial(extend_true, LEFT=LEFT, RIGHT=0)
    nan_appeared = np.apply_along_axis(fn, axis=1, arr=nan_appeared)
    return nan_appeared
