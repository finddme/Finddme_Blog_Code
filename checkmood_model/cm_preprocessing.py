import numpy as np, pandas as pd, os, re

DATA = "./raw_1/"
files = os.listdir(DATA)

input_data = pd.read_json(DATA+files[0], encoding = 'utf-8')
for name in files[1:10]:
    input_data = input_data.append(pd.read_json(DATA+name, encoding = 'utf-8'))

cleaning = lambda s: re.sub("[^0-9가-힣A-z.?! ]", "", s)

input_data["review"] = [cleaning(review) for review in input_data["review"]]

_input, _label = input_data['review'], input_data['rating']

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Okt()
_input = [tokenizer.morphs(review) for review in _input]

words = []
for sen_seq in _input:
    words+=sen_seq
word_index = {word:i for i,word in enumerate(list(set(words)))}
index_word = {v:k for k,v in word_index.items()}
_input = [[word_index[w] for w in sentence] for sentence in _input]
max_len = 256
_input = pad_sequences(_input, maxlen=max_len, padding='post')

to_categorical = lambda x, vec: [ 1 if (i==int(x//2.1)) else 0 for i,v in enumerate(vec)]
_label = [ to_categorical(rate, [0,0,0,0,0]) for rate in _label]

data = (_input, _label, word_index, index_word)

import pickle
FOLDER_SAVE = "./modified/"
with open(FOLDER_SAVE+'input_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)