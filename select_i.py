import numpy as np

def select_i(data,C,a,b):
    y_all = data[:,-1]
    # in_0_C = a[(a<C)&(a>0)]
    # eq_C = a[a==C]
    # eq_0 = a[a==0]
    c1 = np.argwhere(a==0)[:,1]
    c2 = np.argwhere((a>0)&(a<C))[:,1]
    c3 = np.argwhere(a==C)[:,1]
    for i in c2:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if y_all[i]*y_hat!=1:
            return c2[i]
    for i in c1:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if y_all[i]*y_hat<1:
            return c1[i]
    for i in c3:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if y_all[i]*y_hat>1:
            return c3[i]
    
    return -1