from __future__ import division
import unittest
import numpy as np
# from mock import Mock
from itertools import product
from os.path import dirname, join
import sys
# from utils.datatype_handling import pd_array_convert
sys.path.append(dirname(dirname(__file__)))
# sys.path.append(dirname(dirname(dirname(__file__))))
from ops_filter import *

# ROOTDIR = dirname(dirname(dirname(__file__)))
# DATA_PATH = join(ROOTDIR, 'data', 'df.csv')
from data_array import data_array

# Sample data and labels
obj = ['nuclei', 'cytoplasm']
ch = ['DAPI', 'YFP']
prop = ['area', 'mean_intensity', 'min_intensity']
labels = [i for i in product(obj, ch, prop)]
data = np.zeros((len(labels), 10, 5)) # 10 cells, 5 frames
data[:, :, 1:] = 10


class Test_ops_filter(unittest.TestCase):
    def setUp(self):
        pass

    def test_normalize_data(self):
        arr = data_array(data, labels)
        working_arr = arr[0, :, :]
        self.assertEqual(working_arr.max(), 10)
        working_arr = normalize_data(working_arr)
        self.assertEqual(working_arr.max(), 1)
        self.assertEqual(working_arr.min(), 0.0)
        self.assertTrue(np.may_share_memory(working_arr, arr))

    def test_filter_frames_by_range(self):
        arr = data_array(data, labels)
        working_arr = arr[0, :, :]
        self.assertFalse(np.isnan(working_arr[:, 0]).all())
        arr1 = filter_frames_by_range(working_arr, LOWER=0.5, FRAME_START=2)
        self.assertFalse(np.isnan(arr1[:, 0]).all())
        arr2 = filter_frames_by_range(working_arr, LOWER=0.5)
        self.assertTrue(np.isnan(arr2[:, 0]).all())
        self.assertTrue(np.may_share_memory(arr, arr2))

    def test_filter_frames_by_diff(self):
        arr = data_array(data, labels)
        working_arr = arr[0, :, :]
        working_arr[:, 2] = 100
        self.assertFalse(np.isnan(working_arr[:, 1]).all())
        working_arr = filter_frames_by_diff(working_arr, pd_func_name='diff', THRES=0.1, LEFT=1)
        self.assertTrue(np.isnan(working_arr[:, 1]).all())
        self.assertTrue(np.isnan(working_arr[:, 2]).all())
        self.assertTrue(np.may_share_memory(arr, working_arr))

    def def_filter_from_last_frames(self):
        arr = data_array(data, labels)
        working_arr = arr[0, :, :]
        working_arr[5, 3:] = np.nan
        self.assertFalse(np.isnan(working_arr[5, 1]))
        working_arr = filter_from_last_frames(working_arr, LEFT=2)
        self.assertTrue(np.isnan(working_arr[5, 1]))
        self.assertTrue(np.may_share_memory(arr, working_arr))

    def test_interpolate_single_prop(self):
        arr = data_array(data, labels)
        working_arr = arr[0, :, :]
        working_arr[5, 3] = np.nan
        self.assertTrue(np.isnan(working_arr[5, 3]))
        working_arr = interpolate_single_prop(working_arr)
        self.assertFalse(np.isnan(working_arr[5, 3]))
        self.assertTrue(np.may_share_memory(arr, working_arr))

if __name__ == '__main__':
    unittest.main()
