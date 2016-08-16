# from __future__ import division
# import unittest
# import numpy as np
# # from mock import Mock
# from itertools import product
# from os.path import dirname, join, abspath, exists
# import sys
# ROOTDIR = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(dirname(dirname(__file__)))
# print dirname(dirname(__file__))
# from utils.datatype_handling import save_output
# import ops_bool
# from data_array import Site
# from functools import partial
#
#
# # Sample data and labels
# obj = ['nuclei', 'cytoplasm']
# ch = ['DAPI', 'YFP']
# prop = ['area', 'mean_intensity', 'min_intensity']
# labels = [i for i in product(obj, ch, prop)]
# data = np.zeros((len(labels), 10, 5)) # 10 cells, 5 frames
# data[:, :, 1:] = 10
#
# DATA_PATH = join(ROOTDIR, 'data', 'tests.npz')
#
# class Test_ops_filter(unittest.TestCase):
#     def setUp(self):
#         if not exists(DATA_PATH):
#             save_output(data, labels, DATA_PATH)
#
#
#     def test_normalize_data(self):
#         site = Site(dirname(DATA_PATH), 'tests.npz')
#         op = partial(ops_bool.filter_frames_by_range)
#         site.operate(op, pid=1)
#
# if __name__ == '__main__':
#     unittest.main()
