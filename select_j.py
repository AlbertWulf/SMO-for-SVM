import numpy as np
from cal_E import cal_E
def select_j(data,a,i,b):
    y = data[:,-1]
    w = np.dot(np.multiply(a,y).T,data[:,0:-1])
    all_E = np.dot(w,data[:,0:-1].T)-y.T+b
    return np.argmax(np.abs(all_E-cal_E(data,a,i,b)))