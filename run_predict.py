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

# load model
product_model = load_model("model_product_7.h5")
intention_model = load_model("model_intention_7.h5")

# load dicts
word_dict = pickle.load(open('word_dict.bin', "rb"))
category_dict = pickle.load(open('category_dict.bin', "rb"))
intention_dict = pickle.load(open('intention_dict.bin', "rb"))

print(word_dict)
print(category_dict)
print(intention_dict)
