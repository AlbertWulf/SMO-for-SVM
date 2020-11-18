import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm, trange
import operator

def cal_E(data_set,a,i,b):
    y = data_set[:,-1]
    y_hat = np.dot(np.dot(np.multiply(a,y).T,data_set[:,0:-1]),data_set[i,0:-1].T)+b
    return (y_hat-y[i])[0,0]