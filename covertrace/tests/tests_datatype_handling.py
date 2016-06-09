import unittest
import numpy as np
# from mock import Mock
from os.path import dirname, join
import sys
from utils.datatype_handling import pd_array_convert
# sys.path.append(dirname(dirname(__file__)))
# sys.path.append(dirname(dirname(dirname(__file__))))
ROOTDIR = dirname(dirname(dirname(__file__)))
DATA_PATH = join(ROOTDIR, 'data', 'df.csv')


class Test_read_image(unittest.TestCase):
    def setUp(self):
        pass

    def test_pd_array_convert(self):
        pd_array_convert(DATA_PATH)
