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

        self.num_words = min(self.config.MAX_NUM_WORDS, len(self.word2idx_inputs) + 1)

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
        # make sure we do softmax over the time axis
        # expected shape is N x T x D
        # note: the latest version of Keras allows you to pass in axis arg
        def soft_max_axis1(x):
            return softmax(x, axis=1)

        def stack_and_transpose(x):
            # x is a list of length T, each element is a batch_size x output_vocab_size tensor
            x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
            x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
            return x

        def one_step_attention(h, st_1):
            # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
            # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

            # copy s(t-1) Tx times
            # now shape = (Tx, LATENT_DIM_DECODER)
            st_1 = attn_repeat_layer(st_1)

            # Concatenate all h(t)'s with s(t-1)
            # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
            x = attn_concat_layer([h, st_1])

            # Neural net first layer
            x = attn_dense1(x)

            # Neural net second layer with special softmax over time
            alphas = attn_dense2(x)

            # "Dot" the alphas and the h's
            # Remember a.dot(b) = sum over a[t] * b[t]
            context = attn_dot([alphas, h])

            return context
            
        # create embedding layer
        embedding_layer = Embedding(
                            self.num_words,
                            self.config.EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=self.max_len_input,
                            # trainable=True
                            )
        ##### build the model #####

        # Set up the encoder - simple!
        encoder_inputs_placeholder = Input(shape=(self.max_len_input,))
        x = embedding_layer(encoder_inputs_placeholder)
        encoder = Bidirectional(LSTM(self.config.LATENT_DIM,
                                    return_sequences=True,
                                    # dropout=0.5 # dropout not available on gpu
                                    ))
        encoder_outputs = encoder(x)

        # Set up the decoder - not so simple
        decoder_inputs_placeholder = Input(shape=(self.max_len_target,))

        # this word embedding will not use pre-trained vectors
        # although you could
        decoder_embedding = Embedding(self.num_words_output, self.config.EMBEDDING_DIM)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
        
        ######### Attention #########
        # Attention layers need to be global because
        # they will be repeated Ty times at the decoder
        attn_repeat_layer = RepeatVector(self.max_len_input)
        attn_concat_layer = Concatenate(axis=-1)
        attn_dense1 = Dense(10, activation='tanh')
        attn_dense2 = Dense(1, activation=soft_max_axis1)
        attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]

        # define the rest of the decoder (after attention)
        decoder_lstm = LSTM(self.config.LATENT_DIM_DECODER, return_state=True)
        decoder_dense = Dense(self.num_words_output, activation='softmax')

        initial_s = Input(shape=(self.config.LATENT_DIM_DECODER,), name='s0')
        initial_c = Input(shape=(self.config.LATENT_DIM_DECODER,), name='c0')
        context_last_word_concat_layer = Concatenate(axis=2)


        # Unlike previous seq2seq, we cannot get the output
        # all in one step
        # Instead we need to do Ty steps
        # And in each of those steps, we need to consider
        # all Tx h's

        # s, c will be re-assigned in each iteration of the loop
        s = initial_s
        c = initial_c

        # collect outputs in a list at first
        outputs = []
        for t in range(self.max_len_target): # Ty times
            # get the context using attention
            context = one_step_attention(encoder_outputs, s)

            # we need a different layer for each time step
            selector = Lambda(lambda x: x[:, t:t+1])
            xt = selector(decoder_inputs_x)
            
            # combine 
            decoder_lstm_input = context_last_word_concat_layer([context, xt])

            # pass the combined [context, last word] into the LSTM
            # along with [s, c]
            # get the new [s, c] and output
            o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

            # final dense layer to get next word prediction
            decoder_outputs = decoder_dense(o)
            outputs.append(decoder_outputs)
        
        # 'outputs' is now a list of length Ty
        # each element is of shape (batch size, output vocab size)
        # therefore if we simply stack all the outputs into 1 tensor
        # it would be of shape T x N x D
        # we would like it to be of shape N x T x D

        # make it a layer
        stacker = Lambda(stack_and_transpose)
        outputs = stacker(outputs)

        # create the model
        self.model = Model(
                        inputs=[
                        encoder_inputs_placeholder,
                        decoder_inputs_placeholder,
                        initial_s, 
                        initial_c,
                        ],
                        outputs=outputs
                    )

        # compile the model
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        # train the model
        z = np.zeros((self.config.NUM_SAMPLES, self.config.LATENT_DIM_DECODER)) # initial [s, c]
        r = self.model.fit(
                        [self.encoder_inputs, self.decoder_inputs, z, z], self.decoder_targets_one_hot,
                        batch_size=self.config.BATCH_SIZE,
                        epochs=self.config.EPOCHS,
                        validation_split=0.2
                        )

    def save_model(self, SAVE_PATH):
        self.model.save(SAVE_PATH)

    def predict_build_model(self, LOAD_PATH):
        """
        fix model loading part
        """
        # load model
        def soft_max_axis1(x):
            return softmax(x, axis=1)

        def one_step_attention(h, st_1):
            # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
            # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

            # copy s(t-1) Tx times
            # now shape = (Tx, LATENT_DIM_DECODER)
            st_1 = attn_repeat_layer(st_1)

            # Concatenate all h(t)'s with s(t-1)
            # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
            x = attn_concat_layer([h, st_1])

            # Neural net first layer
            x = attn_dense1(x)

            # Neural net second layer with special softmax over time
            alphas = attn_dense2(x)

            # "Dot" the alphas and the h's
            # Remember a.dot(b) = sum over a[t] * b[t]
            context = attn_dot([alphas, h])

            return context

        context_last_word_concat_layer = Concatenate(axis=2)
        attn_repeat_layer = RepeatVector(self.max_len_input)
        attn_concat_layer = Concatenate(axis=-1)
        attn_dense1 = Dense(10, activation='tanh')
        attn_dense2 = Dense(1, activation=soft_max_axis1)
        attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]
        
        t = 0

        self.model = load_model(LOAD_PATH,
                                custom_objects={'soft_max_axis1': soft_max_axis1, 't': t})

        encoder_inputs_placeholder = self.model.input[0]
        encoder_outputs = self.model.layers[3].output
        decoder_embedding = self.model.layers[9]

        initial_s = self.model.layers[2].output
        initial_c = self.model.layers[13].output

        decoder_lstm = self.model.layers[14]
        decoder_dense = self.model.layers[24]

        self.encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

        # next we define a T=1 decoder model
        encoder_outputs_as_input = Input(shape=(self.max_len_input, self.config.LATENT_DIM * 2,))
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        # no need to loop over attention steps this time because there is only one step
        context = one_step_attention(encoder_outputs_as_input, initial_s)

        # combine context with last word
        decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

        # lstm and final dense
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
        decoder_outputs = decoder_dense(o)

        self.decoder_model = Model(
                        inputs=[
                            decoder_inputs_single,
                            encoder_outputs_as_input,
                            initial_s, 
                            initial_c
                        ],
                        outputs=[decoder_outputs, s, c]
                        )

    def decode_sequence(self, input_seqs):
        # Encode the input as state vectors.

        enc_outs = self.encoder_model.predict(input_seqs)
        # Generate empty target sequence of length 1.
        target_seqs = np.zeros((self.config.PREDICTION_BATCH_SIZE, 1))

        # Populate the first character of target sequence with the start character.
        # NOTE: tokenizer lower-cases all words
        for i in range(self.config.PREDICTION_BATCH_SIZE):
            target_seqs[i, 0] = self.word2idx_outputs['<sos>']


        # if we get this we break
        eos = self.word2idx_outputs['<eos>']

        # [s, c] will be updated in each loop iteration
        s = np.zeros((self.config.PREDICTION_BATCH_SIZE, self.config.LATENT_DIM_DECODER))
        c = np.zeros((self.config.PREDICTION_BATCH_SIZE, self.config.LATENT_DIM_DECODER))

        # Create the translation
        output_sentences = [[] for _ in range(self.config.PREDICTION_BATCH_SIZE)]

        for _ in range(self.max_len_target):
            o, s, c = self.decoder_model.predict([target_seqs, enc_outs, s, c])
                
            # Get next word
            idxs = np.argmax(o, axis=1)

            # End sentence of EOS            
            if sum(idxs == eos) == len(idxs):
                break

            for i in range(self.config.PREDICTION_BATCH_SIZE):
                word = ''
                if (idxs[i] > 0) and (idxs[i] !=eos):
                    word = self.idx2word_trans[idxs[i]]
                    output_sentences[i].append(word)
                target_seqs[i, 0] = idxs[i]

        return output_sentences

    def predict(self, input_texts, target_texts):
        # map indexes back into real words
        # so we can view the results

        while True:
            # Do some test translations
            i = np.random.choice(len(input_texts)-self.config.PREDICTION_BATCH_SIZE+1)
            input_seqs = self.encoder_inputs[i:i+self.config.PREDICTION_BATCH_SIZE]
            translations = self.decode_sequence(input_seqs)
            
            for j in range(self.config.PREDICTION_BATCH_SIZE):
                print('-')
                print('Input:', input_texts[i+j])
                print('Translation:', ' '.join(translations[j]))
                print('Actual translation:', target_texts[i+j])

            ans = input("Continue? [Y/n]")
            if ans and ans.lower().startswith('n'):
                break
