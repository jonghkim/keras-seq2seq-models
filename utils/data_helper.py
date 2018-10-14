from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os, sys

def read_txt(DATA_PATH, NUM_SAMPLES):
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
        if t > NUM_SAMPLES:
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

    return input_text, target_texts, target_texts_inputs

def create_vocab(input_texts, target_texts, target_texts_inputs, MAX_NUM_WORDS):
    # tokenize the inputs
    tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_inputs.fit_on_texts(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

    # get the word to index mapping for input language
    word2idx_inputs = tokenizer_inputs.word_index
    print('Found %s unique input tokens.' % len(word2idx_inputs))

    # determine maximum length input sequence
    max_len_input = max(len(s) for s in input_sequences)

    # tokenize the outputs
    # don't filter out special characters
    # otherwise <sos> and <eos> won't appear
    tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
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
    max_len_target = max(len(s) for s in target_sequences)

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

def load_word2vec(WORD2VEC_PATH):
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

def create_embedding_matrix(word2vec, word2idx_inputs, MAX_NUM_WORDS):
    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx_inputs.items():
        if i < MAX_NUM_WORDS:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix