import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import VanillaConfig
from models.seq2seq_model import Seq2SeqModel

if __name__ == "__main__":

    config = VanillaConfig()
    DATA_PATH = 'toy_data/translation/kor.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM

    print(config)
    print("Data Path: ", DATA_PATH)
    print("Word2Vec Path: ", WORD2VEC_PATH)

    #### load the data #### 
    data_helper = DataHelper(config)

    input_texts, target_texts, target_texts_inputs = data_helper.read_txt(DATA_PATH)
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
    model = Seq2SeqModel(config)
    model.set_data(encoder_inputs, decoder_inputs, decoder_targets,
                    max_len_input, max_len_target, num_words_output, embedding_matrix)
    #### build model ####
    model.build_model()    
    #### train model ####
    model.train_model()

    #### save model ####


