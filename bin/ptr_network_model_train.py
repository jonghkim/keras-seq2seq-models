import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import PtrNetworkConfig
from models.ptr_network_model import PtrNetworkModel

if __name__ == "__main__":
    config = PtrNetworkConfig()
    X_DATA_PATH = 'toy_data/number_ordering/x.txt' 
    Y_DATA_PATH = 'toy_data/number_ordering/y.txt'

    SAVE_PATH = 'bin/checkpoints/seq2seq_adv_style_model.h5'

    print(config)
    print("Data Path: ", X_DATA_PATH, Y_DATA_PATH)
    print("Save Path: ", SAVE_PATH)

    data_helper = DataHelper(config)

    #### create number ordering data ####
    data_helper.create_txt_number_ordering(X_DATA_PATH, Y_DATA_PATH)

    #### load the data #### 

