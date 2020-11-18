import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm, trange
from bound import bound
from cal_E import cal_E
from encode import Encoder_X,age
from get_ai_list import get_ai_list
from jud_KKT import jud_KKT
from K_linear import K_linear
from select_i import select_i
from select_j import select_j
from loaddata import load_data
from eva_model import eva_model
from eva_ij import eva_ij
import random
path_filename = 'data/bank-additional-full.csv'
#load data
bd_all = load_data(path_filename)
#选取数据的钱10列足够完成分类
new_bd = bd_all[:,[0,1,2,3,4,5,6,7,8,9,-1]].copy()
bd_yes = new_bd[np.argwhere(new_bd[:,-1]==1)[:,0],:]
bd_no = new_bd[np.argwhere(new_bd[:,-1]==-1)[:,0],:]
blance_data = np.mat(np.zeros((4000,11),dtype=np.float32))
test_set = np.mat(np.zeros((4000,11),dtype=np.float32))
for i in range(0,4000):
    if i%2==0:
        blance_data[i,:] = bd_yes[i,:]
        test_set[i,:] = bd_no[i,:]
    else:
        blance_data[i,:] = bd_no[i,:]
        test_set[i,:] = bd_yes[i,:]
bd = blance_data
#initial parameters
a = np.mat(np.zeros((bd.shape[0],1)))
b = 0
iter_index = 0
#设置参数C
C=0.2
label = bd[:,-1]
#设置最大迭代次数
max_iteration = 10
store_old_ailist = []
list_ai = []
isAll = True
list_ai_all = np.zeros((max_iteration))
predict_result =np.zeros((max_iteration))
start_time = time.time()
store_j = []
for iter_process in trange(max_iteration):
    store_old_ailist = list_ai.copy()
    list_ai = get_ai_list(bd,a,b,C)
    list_ai_all[iter_process] = len(list_ai)
    old_all_j = 0
    old_bound_j = 0
    #print(len(list_ai))
    #判断无违反KKT条件数据时迭代结束
    if len(list_ai) == 0:
        print("finished!")
        break
    else :
        #全集遍历
        if isAll:
            for ii in range(0,bd.shape[0]):
                i = random.randint(0,bd.shape[0]-1)
                #i = index_i
                if jud_KKT(bd,a,b,i,C):

                    j = select_j(bd,a,i,b)
                    if j==old_all_j:
                        j = random.randint(0,bd.shape[0]-1)
                    old_all_j = j
                    if eva_ij(i,j,bd,a,b,C,label)==0:
                        continue
                    else:
                        b,a[i],a[j]=eva_ij(i,j,bd,a,b,C,label)
                    isAll = True
        else :
            arg_bound = np.argwhere((a>0)&(a<C))[:,0]
            for i in arg_bound:
                
                if jud_KKT(bd,a,b,i,C):
                    j = select_j(bd,a,i,b)
                    if j==old_bound_j:
                        j = random.randint(0,bd.shape[0]-1)
                    old_bound_j = j
                    store_j.append(j)
                    if eva_ij(i,j,bd,a,b,C,label)==0:
                        continue
                    else:
                        b,a[i],a[j]=eva_ij(i,j,bd,a,b,C,label)
                    isAll = True
               
    predict_result[iter_process] = eva_model(a,b,bd,bd)           
end_time = time.time()
print("totally cost {:.2f} S!".format(end_time-start_time))  
print("b:{:.4f}".format(b))
#test = bd[0:4000,:]
test = test_set
print(eva_model(a,b,test,bd))
plt.plot(predict_result)
plt.xlabel("iteration/Round")
plt.ylabel("Accuracy")
plt.title("Accuracy of SMO for SVM based on Linear Kernel C={}".format(C))
plt.savefig("figure/linear_C={}.png".format(C),dpi=300)
plt.show()

