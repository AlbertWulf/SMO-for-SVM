import numpy as np

def jud_KKT(data,a,b,i,C):
    y_all = data[:,-1]
    y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
    temp = y_all[i,0]*y_hat[0,0]
    ai = a[i,0]
    if(temp<1-1e-3 and ai<C) or (temp>1+1e-3 and ai>0):
        return True
    else:
        return False
    # if (temp!=1) & ((ai>0)&(ai<C)) or (temp<1) &(ai==0) or (temp>1)&(ai==C):
    #     return True
    # else:
    #     return False