import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

## 작은 데이터(원래, 여기)
def load_data():
    with open('./modified/input_data.pickle', 'rb') as handle:
        return pickle.load(handle)


_input, _label, word_index, index_word = load_data()
max_features = len(word_index)+1
latent_dim = 256

input_train, input_test, label_train, label_test = train_test_split(_input, _label, test_size=.2)


embedding_size = 256

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, embedding_size),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, activation=tf.nn.tanh, return_sequences=True, recurrent_initializer='glorot_uniform')),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

import numpy as np

model.fit(np.array(input_train), np.array(label_train), batch_size=16, epochs=50)
# model.save_weights("./weights/model_weights_1.hdf5")
model.save_weights("./weights/model_weights_3.hdf5")


model.load_weights("./weights/model_weights_3.hdf5")

import re, numpy as np
from konlpy.tag import Okt
cleaning = lambda s: re.sub("[^0-9가-힣A-z.?! ]", "", s)
tokenizer = Okt()

def sentence2numseq(sentence, word_index):
    wordSeq = tokenizer.morphs(cleaning(sentence))
    for i,w in enumerate(wordSeq):
        try:
            wordSeq[i] = word_index[w]
        except:
            wordSeq[i] = 0
    return wordSeq

result = lambda out,idx: idx[list(out).index(max(out))]
result_idx = ["아주 우울","우울","보통","좋음","아주 좋음"]

while True:
    _input = input("하고 싶은 말을 적어보세요.:").strip()
    _input = [sentence2numseq(_input, word_index)]
    print(result(model.predict(_input)[0],result_idx))