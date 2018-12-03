import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import AdvStyleConfig
from models.seq2seq_adv_style_model import Seq2SeqAdvStyleModel

import pandas as pd

if __name__ == "__main__":

    config = AdvStyleConfig()
    NEG_DATA_PATH = 'toy_data/sentiment/neg.txt' 
    POS_DATA_PATH = 'toy_data/sentiment/pos.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM
    LOAD_PATH = 'bin/checkpoints/seq2seq_adv_style_model.h5'
    SAVE_RESULT_PATH ='results/sample_result.csv'

    config.MODE = 'inference'
    
    print(config)

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
    model.predict_sample(input_texts, target_texts)

    ans = raw_input("Save Predictions? [Y/n]")
    if ans and ans.lower().startswith('n'):
        sys.exit("**** Evaluation Done ****")

    translations_results = model.predict(input_texts, target_texts)
    translations_results = [' '.join(translation) for translation in translations_results]

    if config.STYLE_TRANSFER == False:
        compare_list = [list(line) for line in zip(input_texts, target_texts, translations_results)]
        compare_df = pd.DataFrame(compare_list, columns = ["source","target","translated"])
        compare_df.to_csv(SAVE_RESULT_PATH, encoding='utf-8')

    elif config.STYLE_TRANSFER == True:
        failure = translations_results[0::2]
        success = translations_results[1::2]
        compare_list = [list(line) for line in zip(input_texts, target_texts, styles, failure, success)]
        compare_df = pd.DataFrame(compare_list, columns = ["source","target", "style","failure","success"])
        compare_df.to_csv(SAVE_RESULT_PATH, encoding='utf-8')
            
    print("**** Evaluation Done ****")