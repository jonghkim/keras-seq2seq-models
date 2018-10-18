import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import VanillaConfig
from models.seq2seq_adv_style_model import Seq2SeqAdvStyleModel

if __name__ == "__main__":

    config = VanillaConfig()
    NEG_DATA_PATH = 'toy_data/sentiment/neg.txt' 
    POS_DATA_PATH = 'toy_data/sentiment/pos.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM
    LOAD_PATH = 'bin/checkpoints/seq2seq_adv_style_model.h5'

    config.MODE = 'inference'
    
    print(str(config))

    print("Data Path: ", NEG_DATA_PATH, POS_DATA_PATH)
    print("Word2Vec Path: ", WORD2VEC_PATH)
    print("Save Path: ", LOAD_PATH)

    data_helper = DataHelper(config)

    #### load the data #### 
    input_texts, target_texts, target_texts_inputs, styles = data_helper.read_txt_sentiment(NEG_DATA_PATH, POS_DATA_PATH)

    #### tokenize the inputs, outputs ####
    encoder_inputs, decoder_inputs, decoder_targets, \
                word2idx_inputs, word2idx_outputs, \
                max_len_input, max_len_target, num_words_output = \
                            data_helper.create_vocab(input_texts, target_texts, target_texts_inputs)

    #### set data of model ####
    model = Seq2SeqAdvStyleModel(config)
    model.set_data(encoder_inputs, decoder_inputs, decoder_targets, styles,
                    max_len_input, max_len_target, num_words_output, word2idx_inputs, word2idx_outputs)

    model.predict_build_model(LOAD_PATH)
    model.predict(input_texts, target_texts)