import numpy as np
import keras
import powerai_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import sys

import simplejson as json

# from keras.models import load_model

# load model
# product_model = load_model("model_product_7.h5")
# intention_model = load_model("model_intention_7.h5")

# load dicts
word_dict = {}
category_dict = {}
intention_dict = {}

with open('word_dict.json', encoding='utf-8') as data_file:
    word_dict = json.load(data_file)
with open('category_dict.json') as data_file:
    category_dict = json.load(data_file)
with open('intention_dict.json') as data_file:
    intention_dict = json.load(data_file)

print(word_dict)
print(category_dict)
print(intention_dict)
