import os, sys
import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.utils import to_categorical
from keras.models import load_model

#import matplotlib.pyplot as plt
#%matplotlib inline

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from models.config import VanillaConfig

class Seq2SeqModel():

    def __init__(self, config=VanillaConfig()):
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
            self.config.LATENT_DIM,
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
        decoder_embedding = Embedding(self.num_words_output, self.config.LATENT_DIM)
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

        print('---- Build Model Finished ----')
    
    def train_model(self):

        r = self.model.fit(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_targets_one_hot,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_split=0.2,
        )

        print('---- Train Model Finished ----')

    def save_model(self, SAVE_PATH):
        # Save model
        self.model.save(SAVE_PATH)

    def predict_build_model(self, LOAD_PATH):
        # load model
        self.model = load_model(LOAD_PATH)

        encoder_inputs_placeholder = self.model.input[0]
        encoder_outputs, h, c = self.model.layers[4].output
        encoder_states = [h, c]
        decoder_embedding = self.model.layers[3]
        decoder_lstm = self.model.layers[5]
        decoder_dense = self.model.layers[6]        

        # we need to create another model
        # that can take in the RNN state and previous word as input
        # and accept a T=1 sequence.

        # The encoder will be stand-alone
        # From this we will get our initial decoder hidden state
        
        ##### build the model #####
        self.encoder_model = Model(encoder_inputs_placeholder, encoder_states)

        decoder_state_input_h = Input(shape=(self.config.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.config.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # decoder_states_inputs = [decoder_state_input_h] # gru

        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        # this time, we want to keep the states too, to be output
        # by our sampling model
        decoder_outputs, h, c = decoder_lstm(
                                decoder_inputs_single_x,
                                initial_state=decoder_states_inputs
                                )
        # decoder_outputs, state_h = decoder_lstm(
        #   decoder_inputs_single_x,
        #   initial_state=decoder_states_inputs
        # ) #gru
        decoder_states = [h, c]
        # decoder_states = [h] # gru
        decoder_outputs = decoder_dense(decoder_outputs)

        # The sampling model
        # inputs: y(t-1), h(t-1), c(t-1)
        # outputs: y(t), h(t), c(t)
        self.decoder_model = Model(
                            [decoder_inputs_single] + decoder_states_inputs, 
                            [decoder_outputs] + decoder_states
                            )

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        # NOTE: tokenizer lower-cases all words
        target_seq[0, 0] = self.word2idx_outputs['<sos>']

        # if we get this we break
        eos = self.word2idx_outputs['<eos>']

        # Create the translation
        output_sentence = []
        for _ in range(self.max_len_target):
            output_tokens, h, c = self.decoder_model.predict(
                                    [target_seq] + states_value
                                    )
            # output_tokens, h = decoder_model.predict(
            #     [target_seq] + states_value
            # ) # gru

            # Get next word
            idx = np.argmax(output_tokens[0, 0, :])

            # End sentence of EOS
            if eos == idx:
                break

            word = ''
            if idx > 0:
                word = self.idx2word_trans[idx]
                output_sentence.append(word)

            # Update the decoder input
            # which is just the word just generated
            target_seq[0, 0] = idx

            # Update states
            states_value = [h, c]
            # states_value = [h] # gru

        return ' '.join(output_sentence)

    def predict(self, input_texts):
        # map indexes back into real words
        # so we can view the results

        while True:
            # Do some test translations
            i = np.random.choice(len(input_texts))
            input_seq = self.encoder_inputs[i:i+1]
            translation = self.decode_sequence(input_seq)
            print('-')
            print('Input:', input_texts[i])
            print('Translation:', translation)

            ans = input("Continue? [Y/n]")
            if ans and ans.lower().startswith('n'):
                break