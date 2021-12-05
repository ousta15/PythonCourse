from linearReg import linearReg
import numpy as np


X = np.array([[3,4],[6,7],[4,6],[7,3],[3,7]])

Y = np.array([4,5,3,8,9])

b = linearReg(X,Y)

print(b)
