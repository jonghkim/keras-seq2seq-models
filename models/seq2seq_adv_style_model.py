import os, sys
import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Activation, multiply, Concatenate
from keras.utils import to_categorical
from keras.models import load_model
from keras.engine.topology import Layer
from keras.initializers import RandomNormal

import keras.backend as K

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from models.config import AdvStyleConfig

class Seq2SeqAdvStyleModel():
    def __init__(self, config=AdvStyleConfig()):
        assert config.MODE in ["train", "eval", "inference"]
        self.train_phase = config.MODE == "train"
        self.config = config

    def set_data(self, encoder_inputs, decoder_inputs, decoder_targets, styles,
                    max_len_input, max_len_target, num_words_output, word2idx_inputs, word2idx_outputs):

        self.encoder_inputs =encoder_inputs
        self.decoder_inputs =decoder_inputs
        self.decoder_targets = decoder_targets

        self.styles = styles

        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        
        self.num_words_output = num_words_output

        self.word2idx_inputs = word2idx_inputs
        self.word2idx_outputs = word2idx_outputs

        self.idx2word_eng = {v:k for k, v in self.word2idx_inputs.items()}
        self.idx2word_trans = {v:k for k, v in self.word2idx_outputs.items()}

        self.steps_per_epoch = int(len(self.encoder_inputs)/self.config.BATCH_SIZE)

        print('---- Set Data Finished ----')

    def batch_get_input(self, idx, batch_size):
        batch_encoder_inputs = self.encoder_inputs[idx:idx+batch_size]
        batch_decoder_inputs = self.decoder_inputs[idx:idx+batch_size]

        if self.config.ADVERSARIAL == False:
            batch_x = [batch_encoder_inputs, batch_decoder_inputs]

        elif self.config.ADVERSARIAL == True:
            if self.config.STYLE_TRANSFER ==False:
                batch_x = [batch_encoder_inputs, batch_decoder_inputs]

            elif self.config.STYLE_TRANSFER ==True:
                # assign the values

                batch_style_inputs = np.zeros((
                                                batch_size,
                                                self.max_len_target,
                                            ),
                                        dtype='float32'
                                        )

                for i, d in enumerate(self.styles[idx:idx+batch_size]):
                    batch_style_inputs[i,:] = d        

                batch_x = [batch_encoder_inputs, batch_decoder_inputs, batch_style_inputs]

        return batch_x
                
    def batch_get_output(self, idx, batch_size):
        # create targets, since we cannot use sparse
        # categorical cross entropy when we have sequences
        batch_decoder_targets_one_hot = np.zeros((
                                                    batch_size,
                                                    self.max_len_target,
                                                    self.num_words_output
                                                ),
                                                dtype='float32'
                                            )
        # assign the values
        for i, d in enumerate(self.decoder_targets[idx:idx+batch_size]):
            for t, word in enumerate(d):
                batch_decoder_targets_one_hot[i, t, word] = 1

        if self.config.ADVERSARIAL == False:
            batch_y = batch_decoder_targets_one_hot
            
        elif self.config.ADVERSARIAL == True:
            batch_style_targets_one_hot = np.zeros((
                                                batch_size,
                                                self.config.STYLE_NUM
                                            ),
                                        dtype='float32'    
                                        )

            for i, d in enumerate(self.styles[idx:idx+batch_size]):
                batch_style_targets_one_hot[i, d] = 1

            batch_zero_like_loss = np.random.randn(batch_size,1)

            if self.config.STYLE_TRANSFER ==False:
                batch_y = [batch_decoder_targets_one_hot, batch_style_targets_one_hot, batch_zero_like_loss]

            elif self.config.STYLE_TRANSFER ==True:
                batch_y = [batch_decoder_targets_one_hot, batch_style_targets_one_hot, batch_zero_like_loss]

        return batch_y

    def batch_generator(self):
        idx = 0
        max_idx = len(self.encoder_inputs)

        while True:
            if (idx+self.config.BATCH_SIZE) > max_idx:
                idx =0

            batch_size = len(self.encoder_inputs[idx:idx+self.config.BATCH_SIZE])
            batch_x = self.batch_get_input(idx, batch_size)
            batch_y = self.batch_get_output(idx, batch_size)
            idx = idx + batch_size
            yield(batch_x, batch_y)

    def set_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        print('---- Set Embedding Matrix Finished ----')

    def build_encoder(self):
        # create embedding layer
        embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                    self.config.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_len_input,
                                    name='encoder_embedding'
                                    # trainable=True
                                    )

        ##### build the model #####
        encoder_inputs_placeholder = Input(shape=(self.max_len_input,), name='encoder_input')
        x = embedding_layer(encoder_inputs_placeholder)
        encoder = LSTM(
            self.config.LATENT_DIM,
            return_state=True,
            name='encoder_lstm'
            # dropout=0.5 # dropout not available on gpu
        )

        encoder_outputs, h, c = encoder(x)
        # encoder_outputs, h = encoder(x) #gru

        # keep only the states to pass into decoder
        encoder_states = [h, c]
        # encoder_states = [state_h] # gru

        # classifier layer
        if self.config.ADVERSARIAL == True:
            classifier_dense = Dense(self.config.STYLE_NUM, activation='softmax', name='classifier_dense')
            classifier_outputs = classifier_dense(encoder_outputs)

            class AdversarialLoss(Layer):
                def __init__(self, **kwargs):
                    super(AdversarialLoss, self).__init__(**kwargs)

                def call(self, x, mask=None):
                    classifier_outputs = x[0]
                    log_classifier_outputs = K.log(classifier_outputs)
                    
                    adv_loss = multiply([classifier_outputs, log_classifier_outputs])
                    sum_adv_loss = K.sum(adv_loss)
                    self.add_loss(sum_adv_loss,x)
                    
                    return adv_loss

                def get_output_shape_for(self, input_shape):
                    return (input_shape[0][0],1)

            adv_loss = AdversarialLoss()([classifier_outputs])

            return encoder_inputs_placeholder, encoder_states, classifier_outputs, adv_loss

        return encoder_inputs_placeholder, encoder_states

    def build_decoder(self, encoder_states):
        # Set up the decoder, using [h, c] as initial state.
        decoder_inputs_placeholder = Input(shape=(self.max_len_target,), name='decoder_input')

        # this word embedding will not use pre-trained vectors
        # although you could
        decoder_embedding = Embedding(self.num_words_output, self.config.LATENT_DIM, name='decoder_embedding')
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        if self.config.STYLE_TRANSFER ==False:
            # since the decoder is a "to-many" model we want to have
            # return_sequences=True
            decoder_lstm = LSTM(self.config.LATENT_DIM,
                                return_sequences=True,
                                return_state=True,
                                name='decoder_lstm'
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
            decoder_dense = Dense(self.num_words_output, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)

            return decoder_inputs_placeholder, decoder_outputs

        elif self.config.STYLE_TRANSFER ==True:
            # Set up style inputs for the training stage
            # we will give style information in bulks, so that we need same as the max_len_target
            style_inputs_placeholder = Input(shape=(self.max_len_target, ), name='style_input')            
            style_embedding = Embedding(self.config.STYLE_NUM, self.config.STYLE_DIM,
                                        embeddings_initializer= RandomNormal(mean=0.0, stddev=0.05, seed=101), name='style_embedding')    
            style_embedding_x = style_embedding(style_inputs_placeholder)
            
            # Set up the decoder, using [h, c] as initial state.
            decoder_inputs_placeholder = Input(shape=(self.max_len_target,),name='decoder_input')

            # this word embedding will not use pre-trained vectors
            # although you could
            decoder_embedding = Embedding(self.num_words_output, self.config.LATENT_DIM, name='decoder_embedding')
            decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
            
            style_context = Concatenate(axis=-1, name='concatenate')
            style_context_x = style_context([style_embedding_x, decoder_inputs_x])
            
            # since the decoder is a "to-many" model we want to have
            # return_sequences=True
            decoder_lstm = LSTM(self.config.LATENT_DIM,
                                return_sequences=True,
                                return_state=True,
                                name='decoder_lstm'
                                # dropout=0.5 # dropout not available on gpu
                                )
            decoder_outputs, _, _ = decoder_lstm(style_context_x,
                                                initial_state=encoder_states
                                                )

            # decoder_outputs, _ = decoder_gru(
            #   decoder_inputs_x,
            #   initial_state=encoder_states
            # )

            # final dense layer for predictions
            decoder_dense = Dense(self.num_words_output, activation='softmax',name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)

            return decoder_inputs_placeholder, style_inputs_placeholder, decoder_outputs

    def build_model(self):
        if self.config.ADVERSARIAL == False:
            encoder_inputs_placeholder, encoder_states = self.build_encoder()
            decoder_inputs_placeholder, decoder_outputs = self.build_decoder(encoder_states)
            # Create the model object
            model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

            # Compile the model and train it
            model.compile(
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        elif self.config.ADVERSARIAL == True:
            if self.config.STYLE_TRANSFER ==False:
                encoder_inputs_placeholder, encoder_states, classifier_outputs, adv_loss = self.build_encoder()
                decoder_inputs_placeholder, decoder_outputs = self.build_decoder(encoder_states)
                
                model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder],
                                [decoder_outputs, classifier_outputs, adv_loss])
            elif self.config.STYLE_TRANSFER ==True:
                encoder_inputs_placeholder, encoder_states, classifier_outputs, adv_loss = self.build_encoder()
                decoder_inputs_placeholder, style_inputs_placeholder, decoder_outputs = self.build_decoder(encoder_states)
                
                model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder, style_inputs_placeholder],
                  [decoder_outputs, classifier_outputs, adv_loss])

            def zero_loss(y_true, y_pred):
                return K.zeros_like(y_pred)
            
            model.compile(
                optimizer = 'rmsprop',    
                loss = [K.categorical_crossentropy, K.categorical_crossentropy, zero_loss]
            )

        self.model = model

        print('---- Build Model Finished ----')
    
    def train_model(self):
        if self.config.ADVERSARIAL == False:
            r = self.model.fit_generator(generator=self.batch_generator(),
                    epochs=self.config.EPOCHS,
                    steps_per_epoch=self.steps_per_epoch,
                    verbose=1,                    
                    use_multiprocessing=False,
                    workers=1)

        elif self.config.ADVERSARIAL == True:
            if self.config.STYLE_TRANSFER ==False:
                r = self.model.fit_generator(generator=self.batch_generator(),
                        epochs=self.config.EPOCHS,
                        steps_per_epoch=self.steps_per_epoch,
                        verbose=1,                    
                        use_multiprocessing=False,
                        workers=1)
            elif self.config.STYLE_TRANSFER ==True:
                r = self.model.fit_generator(generator=self.batch_generator(),
                    epochs=self.config.EPOCHS,
                    steps_per_epoch=self.steps_per_epoch,
                    verbose=1,                    
                    use_multiprocessing=False,
                    workers=1)
                                
        print('---- Train Model Finished ----')

    def save_model(self, SAVE_PATH):
        # Save model
        self.model.save(SAVE_PATH)

    def load_encoder(self):

        if self.config.ADVERSARIAL == False:
            encoder_inputs_placeholder = self.model.get_layer('encoder_input').output
            encoder_outputs, h, c = self.model.get_layer('encoder_lstm').output
            
            encoder_states = [h, c]
            return encoder_inputs_placeholder, encoder_states

        elif self.config.ADVERSARIAL == True:
            encoder_inputs_placeholder = self.model.get_layer('encoder_input').output
            encoder_outputs, h, c = self.model.get_layer('encoder_lstm').output

            encoder_states = [h, c]

            return encoder_inputs_placeholder, encoder_states
            
    def load_decoder(self):
        decoder_embedding = self.model.get_layer('decoder_embedding')
        decoder_lstm = self.model.get_layer('decoder_lstm')
        decoder_dense = self.model.get_layer('decoder_dense')

        decoder_state_input_h = Input(shape=(self.config.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.config.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # decoder_states_inputs = [decoder_state_input_h] # gru

        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        if self.config.STYLE_TRANSFER == False:
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

            return decoder_inputs_single, decoder_states_inputs, decoder_outputs, decoder_states
        elif self.config.STYLE_TRANSFER == True:
            concatenate = self.model.get_layer('concatenate')
            style_embedding = self.model.get_layer('style_embedding')

            style_inputs_single = Input(shape=(1,))
            style_inputs_single_x = style_embedding(style_inputs_single)

            style_context_x = concatenate([style_inputs_single_x, decoder_inputs_single_x])
            decoder_outputs, h, c = decoder_lstm(
                                    style_context_x,
                                    initial_state=decoder_states_inputs
                                    )
            # decoder_outputs, state_h = decoder_lstm(
            #   decoder_inputs_single_x,
            #   initial_state=decoder_states_inputs
            # ) #gru
            decoder_states = [h, c]
            # decoder_states = [h] # gru
            decoder_outputs = decoder_dense(decoder_outputs)

            return decoder_inputs_single, decoder_states_inputs, style_inputs_single, decoder_outputs, decoder_states
            
    def predict_build_model(self, LOAD_PATH):
        # load model
        class AdversarialLoss(Layer):
            def __init__(self, **kwargs):
                super(AdversarialLoss, self).__init__(**kwargs)

            def call(self, x, mask=None):
                classifier_outputs = x[0]
                log_classifier_outputs = K.log(classifier_outputs)
                
                adv_loss = multiply([classifier_outputs, log_classifier_outputs])
                sum_adv_loss = K.sum(adv_loss)
                self.add_loss(sum_adv_loss,x)
                
                return adv_loss

            def get_output_shape_for(self, input_shape):
                return (input_shape[0][0],1)

        def zero_loss(y_true, y_pred):
            return K.zeros_like(y_pred)

        self.model = load_model(LOAD_PATH, custom_objects={'AdversarialLoss': AdversarialLoss,
                                'zero_loss':zero_loss})

        encoder_inputs_placeholder, encoder_states = self.load_encoder()
        # we need to create another model
        # that can take in the RNN state and previous word as input
        # and accept a T=1 sequence.

        # The encoder will be stand-alone
        # From this we will get our initial decoder hidden state
        
        ##### build the model #####
        self.encoder_model = Model(encoder_inputs_placeholder, encoder_states)
        
        # The sampling model
        # inputs: y(t-1), h(t-1), c(t-1)
        # outputs: y(t), h(t), c(t)
        if self.config.STYLE_TRANSFER == False:
            decoder_inputs_single, decoder_states_inputs, decoder_outputs, decoder_states = self.load_decoder()

            self.decoder_model = Model(
                                [decoder_inputs_single] + decoder_states_inputs, 
                                [decoder_outputs] + decoder_states
                                )
        elif self.config.STYLE_TRANSFER == True:      
            decoder_inputs_single, decoder_states_inputs, style_inputs_single, decoder_outputs, decoder_states = self.load_decoder()

            self.decoder_model = Model(
                                [decoder_inputs_single, style_inputs_single] + decoder_states_inputs, 
                                [decoder_outputs] + decoder_states
                                )

    def decode_sequence(self, input_seqs, input_styles):
        
        # it devised for the end case of input_seqs
        batch_size = len(input_seqs)

        # Generate empty target sequence of length 1.
        if self.config.STYLE_TRANSFER == False:
            # Encode the input as state vectors.            
            states_values = self.encoder_model.predict(input_seqs)
            target_seqs = np.zeros((batch_size, 1))
            # Populate the first character of target sequence with the start character.
            # NOTE: tokenizer lower-cases all words
            for i in range(batch_size):
                target_seqs[i, 0] = self.word2idx_outputs['<sos>']

            # Create the translation
            output_sentences = [[] for _ in range(batch_size)]

        elif self.config.STYLE_TRANSFER == True:
            input_seqs = np.repeat(input_seqs, 2, axis=0)
            input_styles = np.repeat(input_styles, 2, axis=0)

            target_seqs = np.zeros((batch_size*2, 1))
            style_seqs = np.zeros((batch_size*2, 1))

            for i in range(batch_size*2):
                target_seqs[i, 0] = self.word2idx_outputs['<sos>']
                
                if (i%2) == 0:
                    if (input_styles[i][0]%2)==0:
                        style_seqs[i, 0] = input_styles[i][0]
                    else:
                        style_seqs[i, 0] = input_styles[i][0] - 1                        
                else:
                    if (input_styles[i][0]%2)==0:
                        style_seqs[i, 0] = input_styles[i][0] + 1
                    else:
                        style_seqs[i, 0] = input_styles[i][0]

            output_sentences = [[] for _ in range(batch_size*2)]
            states_values = self.encoder_model.predict(input_seqs)

        # if we get this we break
        eos = self.word2idx_outputs['<eos>']
        
        for _ in range(self.max_len_target):
            if self.config.STYLE_TRANSFER == False:
                output_tokens, h, c = self.decoder_model.predict(
                                        [target_seqs] + states_values
                                        )
                # output_tokens, h = decoder_model.predict(
                #     [target_seq] + states_value
                # ) # gru
            elif self.config.STYLE_TRANSFER == True:
                output_tokens, h, c = self.decoder_model.predict(
                                        [target_seqs, style_seqs] + states_values
                                        )
                             
            # Get next word
            idxs = np.argmax(output_tokens, axis=2)
            idxs = idxs.reshape(-1)
            
            ## End sentence of EOS
            if sum(idxs == eos) == len(idxs):
                break

            if self.config.STYLE_TRANSFER == False:
                for i in range(batch_size):
                    word = ''
                    if (idxs[i] > 0) and (idxs[i] !=eos):
                        word = self.idx2word_trans[idxs[i]]
                        output_sentences[i].append(word)
                    else:
                        output_sentences[i].append("<eos>")
                    # Update the decoder input
                    # which is just the word just generated
                    target_seqs[i, 0] = idxs[i]
            elif self.config.STYLE_TRANSFER == True:
                for i in range(batch_size*2):
                    word = ''
                    if (idxs[i] > 0) and (idxs[i] !=eos):
                        word = self.idx2word_trans[idxs[i]]
                        output_sentences[i].append(word)
                    # Update the decoder input
                    # which is just the word just generated
                    else:
                        output_sentences[i].append("<eos>")
                    target_seqs[i, 0] = idxs[i]
            # Update states
            states_values = [h, c]
            # states_value = [h] # gru

        # <eos> masking
        for i, output_sentence in enumerate(output_sentences):
            if "<eos>" in output_sentence:
                eos_idx = output_sentence.index("<eos>")
                output_sentence = output_sentence[:min(eos_idx, self.max_len_target)]
                output_sentences[i] = output_sentence

        return output_sentences

    def predict_sample(self, input_texts, target_texts):
        # map indexes back into real words
        # so we can view the results
        while True:
            # Do some test translations
            i = np.random.choice(len(input_texts)-self.config.PREDICT_SAMPLE_SIZE+1)
            
            batch_size = len(self.encoder_inputs[i:i+self.config.PREDICT_SAMPLE_SIZE])
            batch_encoder_inputs = None
            batch_style_inputs = None

            if self.config.ADVERSARIAL == False:
                [batch_encoder_inputs, _] = self.batch_get_input(i, batch_size)
            elif self.config.ADVERSARIAL == True:
                if self.config.STYLE_TRANSFER ==False:
                   [batch_encoder_inputs, _] = self.batch_get_input(i, batch_size)
                elif self.config.STYLE_TRANSFER ==True:
                    [batch_encoder_inputs, _, batch_style_inputs] = self.batch_get_input(i, batch_size)

            translations = self.decode_sequence(batch_encoder_inputs, batch_style_inputs)

            for j in range(batch_size):
                print('-')
                print('*** Input:', input_texts[i+j], ' ***')

                if self.config.STYLE_TRANSFER == False:
                    print('   Translation:', ' '.join(translations[j]))
                    print('Actual translation:', target_texts[i+j])

                elif self.config.STYLE_TRANSFER == True:
                    for k in range(2):
                        if batch_style_inputs[j][0]%2 ==0:
                            target_style = batch_style_inputs[j][0]+k
                        else:
                            target_style = batch_style_inputs[j][0]-1+k

                        print('   Translation to Style {}:'.format(str(target_style)), 
                                ' '.join(translations[j*2+k]))

                    print('Actual styles:', batch_style_inputs[j][0])
                    print('Actual translation:', target_texts[i+j])

            ans = raw_input("Continue? [Y/n]")
            if ans and ans.lower().startswith('n'):
                break

    def predict(self, input_texts, target_texts):
        
        translations_results = []
        
        step = 0

        for i in range(0, len(input_texts), self.config.PREDICTION_BATCH_SIZE):
            if step < (i/len(input_texts)):
                print("Progrees: ", i," / ", len(input_texts))
                step = step+0.05
            batch_size = len(self.encoder_inputs[i:i+self.config.PREDICTION_BATCH_SIZE])
            batch_encoder_inputs = None
            batch_style_inputs = None

            if self.config.ADVERSARIAL == False:
                [batch_encoder_inputs, _] = self.batch_get_input(i, batch_size)
            elif self.config.ADVERSARIAL == True:
                if self.config.STYLE_TRANSFER ==False:
                   [batch_encoder_inputs, _] = self.batch_get_input(i, batch_size)
                elif self.config.STYLE_TRANSFER ==True:
                    [batch_encoder_inputs, _, batch_style_inputs] = self.batch_get_input(i, batch_size)

            translations = self.decode_sequence(batch_encoder_inputs, batch_style_inputs)
            translations_results.extend(translations) 

        return translations_results    
