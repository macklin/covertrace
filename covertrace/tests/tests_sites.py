from __future__ import division
import unittest
import numpy as np
from mock import Mock
from itertools import product
import os
from os.path import dirname, join, abspath, exists
import sys
ROOTDIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(dirname(dirname(__file__)))
from data_array import Site
from functools import partial
from utils.datatype_handling import save_output
from data_array import Sites

# Sample data and labels
obj = ['nuclei', 'cytoplasm']
ch = ['DAPI', 'YFP']
prop = ['area', 'mean_intensity', 'min_intensity']
labels = [i for i in product(obj, ch, prop)]
labels.append(['abs_id', ])
arr = np.zeros((len(labels), 10, 5))  # 10 cells, 5 frames
arr[:, :, 1:] = 10

DATA_PATH = join(ROOTDIR, 'data', 'tests.npz')

class Test_site(unittest.TestCase):
    def setUp(self):
        if not exists(DATA_PATH):
            save_output(arr, labels, DATA_PATH)

    def test_having_data(self):
        site = Site(dirname(DATA_PATH), 'tests.npz')
        self.assertTrue(isinstance(site.data.arr, np.ndarray))

    def test_save_file(self):
        site = Site(dirname(DATA_PATH), 'tests.npz')
        self.assertFalse(exists(join(site.directory, 'new_tests.npz')))
        site.save(new_file_name='new_tests.npz')
        self.assertTrue(exists(join(site.directory, 'new_tests.npz')))
        os.remove(join(site.directory, 'new_tests.npz'))


class Test_sites(unittest.TestCase):
    def setUp(self):
        self.parent_folder = join(ROOTDIR, 'data', 'sample')
        self.subfolders = ('sample1', 'sample2')
        self.file_name = 'tests.npz'
        for sub in self.subfolders:
            if not exists(join(self.parent_folder, sub)):
                os.makedirs(join(self.parent_folder, sub))
                save_output(arr, labels, join(self.parent_folder, sub, self.file_name))

    def tearDown(self):
        for sub in self.subfolders:
            os.remove(join(self.parent_folder, sub, 'tests.npz'))
            try:
                os.remove(join(self.parent_folder, sub, 'arr_modified.npz'))
            except:
                pass
            os.removedirs(join(self.parent_folder, sub))

    def test_having_data(self):
        sites = Sites(self.parent_folder, self.subfolders, file_name=self.file_name)
        for site in sites:
            self.assertTrue(isinstance(site.data.arr, np.ndarray))
        self.assertEqual(len(sites), 2)

    def test_not_merging(self):
        conditions = ['A', 'B']
        sites = Sites(self.parent_folder, self.subfolders,
                      conditions=conditions, file_name=self.file_name)
        before_cell_sum = sum([i.data.arr.shape[1] for i in sites])
        sites.merge_conditions()
        self.assertFalse(len(sites) == 1)
        self.assertNotEqual(before_cell_sum, sites.sample1.data.arr.shape[1])

    def test_merging(self):
        conditions = ['A', 'A']
        sites = Sites(self.parent_folder, self.subfolders,
                      conditions=conditions, file_name=self.file_name)
        before_cell_sum = sum([i.data.arr.shape[1] for i in sites])
        sites.merge_conditions()
        self.assertTrue(len(sites) == 1)
        self.assertEqual(before_cell_sum, sites.sample1.data.arr.shape[1])

    def tests_collect(self):
        sites = Sites(self.parent_folder, self.subfolders, file_name=self.file_name)
        sites.set_state(['nuclei', 'DAPI', 'area'])
        panels = sites.collect()
        self.assertTrue(isinstance(panels, list))
        self.assertEqual(len(panels), 2)


class Test_site_operate(unittest.TestCase):
    def setUp(self):
        if not exists(DATA_PATH):
            save_output(arr, labels, DATA_PATH)

    def test_having_data(self):
        site = Site(dirname(DATA_PATH), 'tests.npz')
        import ops_filter
        op = partial(ops_filter.normalize_data)
        site._set_state(['nuclei', 'DAPI', 'area'])
        self.assertEqual(site.data['nuclei', 'DAPI', 'area'].max(), 10.0)
        site.operate(op)
        self.assertEqual(site.data['nuclei', 'DAPI', 'area'].max(), 1.0)


if __name__ == '__main__':
    unittest.main()
