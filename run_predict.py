#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import keras
import powerai_data_binary as powerai_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import sys
from keras.models import load_model
import pickle
import run_predict_data

batch_size=10
# load model
product_model = load_model("model_product_7.h5")
intention_model = load_model("model_intention_7.h5")

# load dicts
word_dict = pickle.load(open('word_dict.bin', "rb"))
category_dict = pickle.load(open('category_dict.bin', "rb"))
intention_dict = pickle.load(open('intention_dict.bin', "rb"))

(id_list, test_a, test_b) = run_predict_data.load_data("./trainval/data/", word_dict)

print(id_list)
print(test_a)
print(test_b)
result_a = product_model.predict(test_a, batch_size=batch_size, verbose=1)
result_b = intention_model.predict(test_b, batch_size=batch_size, verbose=1)


