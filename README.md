# keras-seq2seq-models

*Work-in-Progress*

This is a project to learn different s2s models based on Keras Functional API

## Data
- For seq2seq model & seq2seq attn model: Translation Dataset 
    - get the data at: http://www.manythings.org/anki/
- For adversarial style embedding model: Sentiment Review Dataset
    - get the data at: http://jmcauley.ucsd.edu/data/amazon/links.html

## Models
- Vanilla seq2seq model (Done)
- Seq2seq model with attention mechanism (Done)

- Vanilla seq2seq model with adversarial network and style embedding (Work-in-Progress)

- Pointer network model (Work-in-Progress)
- Pointer-Generator model (Work-in-Progress)

## Usage
~~~
# For vanilla seq2seq model
python -m bin.seq2seq_model_train
python -m bin.seq2seq_model_test

# For seq2seq with attention mechanism model
python -m bin.seq2seq_attn_model_train
python -m bin.seq2seq_attn_model_test

# For vanilla seq2seq model with adversarial network and style embedding (work-in-progress)
python -m bin.seq2seq_adv_style_model_train
python -m bin.seq2seq_adv_style_model_test
~~~

### References
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), 2014.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014.
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368), 2017.
- [Style Transfer in Text: Exploration and Evaluation](https://arxiv.org/abs/1711.06861), 2017.