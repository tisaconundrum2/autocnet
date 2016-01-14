import os
import sys
from time import gmtime, strftime
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from autocnet.control import control


class TestC(unittest.TestCase):

    def setUp(self):
        ids = ['pt1','pt1', 'pt1', 'pt2', 'pt2']
        ptype = [2,2,2,2,2]
        serials = ['a', 'b', 'c', 'b', 'c']
        mtype = [2,2,2,2,2]

        multi_index = pd.MultiIndex.from_tuples(list(zip(ids, ptype, serials, mtype)),
                                    names=['Id', 'Type', 'Serial Number', 'Measure Type'])


        columns = ['Random Number']
        self.data_length = 5
        data = np.random.randn(self.data_length)

        self.C = control.C(data, index=multi_index, columns=columns)

    def test_n_point(self):
        self.assertEqual(self.C.n,2)

    def test_n_measures(self):
        self.assertEqual(self.C.m, self.data_length)

    def test_modified_date(self):
        self.assertEqual(self.C.modifieddate, 'Not modified')

    def test_creation_date(self):
        self.assertEqual(self.C.creationdate, strftime("%Y-%m-%d %H:%M:%S", gmtime()))