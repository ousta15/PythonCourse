import unittest
from linearReg import linearReg
import numpy as np

class LinearRegTest(unittest.TestCase):

    def test_shape(self):
        self.assertTrue("Number of rows of X and y should be the same.", linearReg(X = {'x1': [3,5,5], 'x2': [4,3,6]}, y = {"y":[4]}))

    def test_nan(self):
        self.assertTrue("All rows contain at least one NAN value.", linearReg(X = {'x1': [3,np.nan], 'x2': [np.nan,6]}, y = {"y":[4,np.nan]}))

    def test_type(self):
        self.assertRaises(TypeError,linearReg(X = {'x1': [3,"asdf"], 'x2': [8,6]}, y = {"y":[4,"ghjk"]}))

if __name__ == '__main__':
    unittest.main()
