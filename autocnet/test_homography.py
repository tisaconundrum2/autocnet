import unittest

class TestCalculateHomography(unittest.TestCase):
    def setUp(self):
        self.src_points = ""
        self.des_points = ""

    def test_compute_homography(self):
        self.assertEqual(self.src_points, self.des_points)

    def tearDown(self):
        print(False)
