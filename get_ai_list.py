import numpy as np

def get_ai_list(data,a,b,C):
    list_a = []
    y_all = data[:,-1]
    c1 = np.argwhere(a==0)[:,0]
    c2 = np.argwhere((a>0)&(a<C))[:,0]
    c3 = np.argwhere(a==C)[:,0]
    for i in c2:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if abs(y_all[i,0]*y_hat[0,0]-1)>1e-3:
            list_a.append(i)
    for i in c1:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if y_all[i,0]*y_hat[0,0]<1-1e-3:
            list_a.append(i)
    for i in c3:
        y_hat=np.dot(np.dot(np.multiply(a,y_all).T,data[:,0:-1]),data[i,0:-1].T)+b
        if y_all[i,0]*y_hat[0,0]>1+1e-3:
            list_a.append(i)
    
    return list_a