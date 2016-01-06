import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('..'))

from autocnet.control import control


class TestC(unittest.TestCase):

    def setUp(self):
        self.C = control.C()

    def test_n_point(self):
        self.assertEqual(self.C.n, 100)

    def test_n_measures(self):
        self.assertEqual(self.C.m, 500)