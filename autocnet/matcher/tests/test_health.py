import unittest

from .. import health


class TestEdgeHealth(unittest.TestCase):

    def setUp(self):
        self.H = health.EdgeHealth()

    def test_fundamental(self):
        self.assertEqual(self.H.FundamentalMatrix, 0.0)

    def test_update(self):
        self.H.foo = 'a'
        self.H.bar = 1
        self.H.update(**{'foo': 'b', 'bar': 2})

        self.assertEqual(self.H.foo, 'b')
        self.assertEqual(self.H.bar, 2)