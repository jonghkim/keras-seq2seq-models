import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

#import matplotlib.pyplot as plt
#%matplotlib inline

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from models.config import VanillaConfig

class S2SModel():

    def __init__(self, config=VanillaConfig()):
        assert config.mode in ["train", "eval", "inference"]
        self.train_phase = config.mode == "train"
        self.config = config


    def set_data(self, encoder_inputs, decoder_inputs, decoder_targets,
                    max_len_input, max_len_target, num_words_output, embedding_matrix):

        self.encoder_inputs =encoder_inputs
        self.decoder_inputs =decoder_inputs
        self.decoder_targets = decoder_targets

        self.embedding_matrix = embedding_matrix

        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        
        self.num_words_output = num_words_output

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

        print('Set Data Finished')

    def build_model(self):
        # create embedding layer
        embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                    self.config.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_len_input,
                                    # trainable=True
                                    )


        ##### build the model #####
        encoder_inputs_placeholder = Input(shape=(self.max_len_input,))
        x = embedding_layer(encoder_inputs_placeholder)
            encoder = LSTM(
            LATENT_DIM,
            return_state=True,
            # dropout=0.5 # dropout not available on gpu
        )

        encoder_outputs, h, c = encoder(x)
        # encoder_outputs, h = encoder(x) #gru

        # keep only the states to pass into decoder
        encoder_states = [h, c]
        # encoder_states = [state_h] # gru

        # Set up the decoder, using [h, c] as initial state.
        decoder_inputs_placeholder = Input(shape=(self.max_len_target,))

        # this word embedding will not use pre-trained vectors
        # although you could
        decoder_embedding = Embedding(num_words_output, self.config.LATENT_DIM)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        # since the decoder is a "to-many" model we want to have
        # return_sequences=True
        decoder_lstm = LSTM(
            self.config.LATENT_DIM,
            return_sequences=True,
            return_state=True,
            # dropout=0.5 # dropout not available on gpu
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs_x,
            initial_state=encoder_states
        )

        # decoder_outputs, _ = decoder_gru(
        #   decoder_inputs_x,
        #   initial_state=encoder_states
        # )

        # final dense layer for predictions
        decoder_dense = Dense(self.num_words_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Create the model object
        model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

        # Compile the model and train it
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )   

        self.model = model

        print('Build Model Finished')
    
    def train_model(self):

        r = self.model.fit(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_targets_one_hot,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_split=self.config.VALIDATION_SPLIT,
        )

        print('Train Model Finished')


    def save_model(self, SAVE_PATH):
        # Save model
        self.model.save(SAVE_PATH)
    