import os, sys
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
        Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import numpy as np

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from models.config import AttnConfig

class Seq2SeqAttnModel():
    def __init__(self, config=AttnConfig()):
        assert config.MODE in ["train", "eval", "inference"]
        self.train_phase = config.MODE == "train"
        self.config = config

    def set_data(self, encoder_inputs, decoder_inputs, decoder_targets,
                    max_len_input, max_len_target, num_words_output, word2idx_inputs, word2idx_outputs):

        self.encoder_inputs =encoder_inputs
        self.decoder_inputs =decoder_inputs
        self.decoder_targets = decoder_targets


        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        
        self.num_words_output = num_words_output

        self.word2idx_inputs = word2idx_inputs
        self.word2idx_outputs = word2idx_outputs

        self.idx2word_eng = {v:k for k, v in self.word2idx_inputs.items()}
        self.idx2word_trans = {v:k for k, v in self.word2idx_outputs.items()}

        # create targets, since we cannot use sparse
        # categorical cross entropy when we have sequences
        self.decoder_targets_one_hot = np.zeros((
                                                len(self.encoder_inputs),
                                                self.max_len_target,
                                                self.num_words_output
                                            ),
                                            dtype='float32'
                                        )

        # assign the values
        for i, d in enumerate(self.decoder_targets):
            for t, word in enumerate(d):
                self.decoder_targets_one_hot[i, t, word] = 1

        print('---- Set Data Finished ----')

    def set_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        print('---- Set Embedding Matrix Finished ----')


    def build_model(self):
        pass

    def train_model(self):
        pass        