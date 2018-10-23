import os, sys
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
        Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.activations import softmax

from keras.models import load_model

import keras.backend as K

import numpy as np

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from models.config import PtrNetworkConfig

class PtrNetworkModel():
    def __init__(self, config=PtrNetworkConfig()):
        assert config.MODE in ["train", "eval", "inference"]
        self.train_phase = config.MODE == "train"
        self.config = config
