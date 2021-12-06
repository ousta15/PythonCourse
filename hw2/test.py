import unittest
from linearReg import linearReg

class LinearRegTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual("Number of rows of X and y should be the same.", linearReg(X = [[3,5,5],[4,3,6]], y = [4]))


if __name__ == '__main__':
    unittest.main()
