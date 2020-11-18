import pandas as pd
from encode import Encoder_X,age
import numpy as np
def load_data(path_filename):
    bank = pd.read_csv(path_filename,sep=',')
    encoder_bank = Encoder_X(bank)
    encoder_bank = age(encoder_bank)
    all_value = encoder_bank.values
    bank_data_noDuration = np.delete(all_value,10,axis=1)
    bd = np.mat(bank_data_noDuration)

    return bd