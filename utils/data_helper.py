from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import os, sys

class DataHelper():
    def __init__(self, config):
        self.config = config

    def create_txt_number_ordering(self, X_DATA_PATH, Y_DATA_PATH):
        import numpy as np

        n_examples = 10000
        n_steps = 8

        arange = 10

        x_file = X_DATA_PATH
        y_file = Y_DATA_PATH
        
        # no repeating numbers within a sequence
        x = np.arange( arange ).reshape( 1, -1 ).repeat( n_examples, axis = 0 )
        x = np.apply_along_axis( np.random.permutation, 1, x )
        x = x[:,:n_steps]

        y = np.argsort( x, axis = 1 )

        np.savetxt( x_file, x, delimiter = ',', fmt = '%d' )
        np.savetxt( y_file, y, delimiter = ',', fmt = '%d' )

    def read_txt_translation(self, DATA_PATH):
        # Where we will store the data
        input_texts = [] # sentence in original language
        target_texts = [] # sentence in target language
        target_texts_inputs = [] # sentence in target language offset by 1

        # load in the data
        # download the data at: http://www.manythings.org/anki/
        t = 0
        for line in open(DATA_PATH):
            # only keep a limited number of samples
            t += 1
            if t > self.config.NUM_SAMPLES:
                break

            # input and target are separated by tab
            if '\t' not in line:
                continue

            # split up the input and translation
            input_text, translation = line.rstrip().split('\t')

            # make the target input and output
            # recall we'll be using teacher forcing
            target_text = translation + ' <eos>'
            target_text_input = '<sos> ' + translation

            input_texts.append(input_text)
            target_texts.append(target_text)
            target_texts_inputs.append(target_text_input)

        print("num samples:", len(input_texts))
        
        return input_texts, target_texts, target_texts_inputs

    def read_txt_sentiment(self, NEG_DATA_PATH, POS_DATA_PATH):
        # Where we will store the data
        neg_texts = [] # sentence in original language
        pos_texts = [] # sentence in target language
        styles = []

        # load in the data
        # download the data at: http://www.manythings.org/anki/
        t = 0
        for line in open(NEG_DATA_PATH):
            # only keep a limited number of samples
            t += 1
            if t > self.config.NUM_SAMPLES:
                break
            # split up the input and translation
            neg_text = line.rstrip()
            neg_texts.append(neg_text)
            styles.append(0)
                
        t = 0    
        for line in open(POS_DATA_PATH):
            # only keep a limited number of samples
            t += 1
            if t > self.config.NUM_SAMPLES:
                break
            # split up the input and translation
            pos_text = line.rstrip()
            pos_texts.append(pos_text)
            styles.append(1)
            
        print("num samples:", len(neg_texts)+len(pos_texts))

        input_texts = neg_texts + pos_texts

        target_texts = []
        target_texts_inputs = []

        for input_text in input_texts:
            target_texts.append(input_text+' <eos>')
            target_texts_inputs.append('<sos> ' + input_text)    
        
        return input_texts, target_texts, target_texts_inputs, styles

    def create_vocab(self, input_texts, target_texts, target_texts_inputs):
        # tokenize the inputs
        tokenizer_inputs = Tokenizer(num_words=self.config.MAX_NUM_WORDS)
        tokenizer_inputs.fit_on_texts(input_texts)
        input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

        # get the word to index mapping for input language
        word2idx_inputs = tokenizer_inputs.word_index
        print('Found %s unique input tokens.' % len(word2idx_inputs))

        # determine maximum length input sequence
        if self.config.ENCODER_INPUT_MAX_LEN != None:
            max_len_input = self.config.ENCODER_INPUT_MAX_LEN
        else:
            max_len_input = max(len(s) for s in input_sequences)


        # tokenize the outputs
        # don't filter out special characters
        # otherwise <sos> and <eos> won't appear
        tokenizer_outputs = Tokenizer(num_words=self.config.MAX_NUM_WORDS, filters='')
        tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
        target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
        target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

        # get the word to index mapping for output language
        word2idx_outputs = tokenizer_outputs.word_index
        print('Found %s unique output tokens.' % len(word2idx_outputs))

        # store number of output words for later
        # remember to add 1 since indexing starts at 1
        num_words_output = len(word2idx_outputs) + 1

        # determine maximum length output sequence
        if self.config.ENCODER_INPUT_MAX_LEN != None:
            max_len_target = max(len(s) for s in target_sequences)
        else:
            max_len_target = self.config.DECODER_INPUT_MAX_LEN

        # pad the sequences
        encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
        print("encoder_inputs.shape:", encoder_inputs.shape)
        print("encoder_inputs[0]:", encoder_inputs[0])

        decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
        print("decoder_inputs[0]:", decoder_inputs[0])
        print("decoder_inputs.shape:", decoder_inputs.shape)

        decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

        return (encoder_inputs, decoder_inputs, decoder_targets, word2idx_inputs, 
                word2idx_outputs, max_len_input, max_len_target, num_words_output)

    def load_word2vec(self, WORD2VEC_PATH):
        # store all the pre-trained word vectors
        print('Loading word vectors...')
        word2vec = {}
        with open(WORD2VEC_PATH) as f:
            # is just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print('Found %s word vectors.' % len(word2vec))
        return word2vec

    def create_embedding_matrix(self, word2vec, word2idx_inputs):
        # prepare embedding matrix
        print('Filling pre-trained embeddings...')
        num_words = min(self.config.MAX_NUM_WORDS, len(word2idx_inputs) + 1)
        embedding_matrix = np.zeros((num_words, self.config.EMBEDDING_DIM))
        for word, i in word2idx_inputs.items():
            if i < self.config.MAX_NUM_WORDS:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all zeros.
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix