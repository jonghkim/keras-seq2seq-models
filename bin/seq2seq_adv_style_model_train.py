import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import AdvStyleConfig
from models.seq2seq_adv_style_model import Seq2SeqAdvStyleModel

if __name__ == "__main__":
    config = AdvStyleConfig()
    NEG_DATA_PATH = 'toy_data/sentiment/neg.txt' 
    POS_DATA_PATH = 'toy_data/sentiment/pos.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM
    SAVE_PATH = 'bin/checkpoints/seq2seq_adv_style_model.h5'

    print(config)
    print("Data Path: ", NEG_DATA_PATH, POS_DATA_PATH)
    print("Word2Vec Path: ", WORD2VEC_PATH)
    print("Save Path: ", SAVE_PATH)

    data_helper = DataHelper(config)

    #### load the data #### 
    input_texts, target_texts, target_texts_inputs, styles = data_helper.read_txt_sentiment(NEG_DATA_PATH, POS_DATA_PATH)

    #### tokenize the inputs, outputs ####
    encoder_inputs, decoder_inputs, decoder_targets, \
     word2idx_inputs, word2idx_outputs, \
     max_len_input, max_len_target, num_words_output = \
                         data_helper.create_vocab(input_texts, target_texts, target_texts_inputs)
                         
    #### load word2vec pretrained model ####
    word2vec = data_helper.load_word2vec(WORD2VEC_PATH)

    #### create embedding matrix ####
    embedding_matrix = data_helper.create_embedding_matrix(word2vec, word2idx_inputs)
    
    #### set data of model ####
    model = Seq2SeqAdvStyleModel(config)
    model.set_data(encoder_inputs, decoder_inputs, decoder_targets, styles,
                    max_len_input, max_len_target, num_words_output, word2idx_inputs, word2idx_outputs)
    model.set_embedding_matrix(embedding_matrix)    
    #### build model ####
    model.build_model()    
    #### train model ####
    model.train_model()
    #### save model ####
    model.save_model(SAVE_PATH)
