import numpy as np
def eva_model(a,b,test_set,train_set):
    y = train_set[:,-1]
    test_label = test_set[:,-1].T
    w = np.dot(np.multiply(a,y).T,train_set[:,0:-1])
    yy = np.dot(w,test_set[:,0:-1].T)+b
    yy[np.where(yy>=0)] = 1
    yy[np.where(yy<0)] = -1
    #print(np.argwhere(yy==-1).shape)
    acc = np.sum(np.abs(yy+test_label),axis=1)[0,0]
    return acc/test_set.shape[0]/2