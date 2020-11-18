import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm, trange
import operator

def Encoder_X(data):
    Attribute_Array = np.array(['age','job','marital','education','default','housing','loan','contact','month','day_of_week','campaign','pdays','previous','poutcome'])
    for i in range(0,Attribute_Array.shape[0]):
        data[Attribute_Array[i]].replace(data[Attribute_Array[i]].unique(), np.arange(1,data[Attribute_Array[i]].unique().shape[0]+1), inplace=True)
    data['y'].replace(['yes','no'],[1,-1],inplace=True)
    return data
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe