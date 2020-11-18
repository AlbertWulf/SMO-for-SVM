import numpy as np

def K_linear(x1,x2):
    return np.dot(x1,x2.T)[0,0]