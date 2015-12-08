import unittest


class TestX(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_X(self):
        self.assertEquals(1, 1)

if __name__ == '__main__':
    unittest.main()
