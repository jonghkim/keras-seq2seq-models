# keras-seq2seq-models

*Work-in-Progress*

This is a project to learn different s2s models based on Keras Functional API

## Data
- For seq2seq model & seq2seq attn model: Translation Dataset 
    - get the data at: http://www.manythings.org/anki/
- For adversarial style embedding model: Sentiment Review Dataset
    - get the data at: http://jmcauley.ucsd.edu/data/amazon/links.html
- For pointer network model: Integer Sequence Ordering
    - get the data generation code from: https://github.com/zygmuntz/pointer-networks-experiments/blob/master/generate_data.py

## Models
- vanilla seq2seq model (Done)
- seq2seq model with attention mechanism (Done)
- seq2seq auto-encoder model with adversarial network and style embedding (Done)
- pointer network model (Work-in-Progress)
- pointer-generator model (Work-in-Progress)
- transformer model (Work-in-Progress)
- bert model (Work-in-Progress)

## Usage
~~~
# For vanilla seq2seq model: Solve Translation Problem
python -m bin.seq2seq_model_train
python -m bin.seq2seq_model_test

# For seq2seq with attention mechanism model: Solve Translation Problem
python -m bin.seq2seq_attn_model_train
python -m bin.seq2seq_attn_model_test

# For seq2seq auto-encoder model with adversarial network and style embedding: Solve Style Transfer Problem
python -m bin.seq2seq_adv_style_model_train
python -m bin.seq2seq_adv_style_model_test

# For pointer network model (*Work-in-Progress*): Solve Interger Sequence Ordering Problem
python -m bin.ptr_network_model_train
python -m bin.ptr_network_model_test
~~~

### Code References
- For base seq2seq: [Keras Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
- For adversarial style embedding model: [Theano Implementation of Style Transfer](https://github.com/fuzhenxin/text_style_transfer)

### Paper References
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), 2014.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014.
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368), 2017.
- [Style Transfer in Text: Exploration and Evaluation](https://arxiv.org/abs/1711.06861), 2017.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), 2018.
