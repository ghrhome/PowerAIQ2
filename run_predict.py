#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import numpy as np
# import keras
# import powerai_data
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.preprocessing.text import Tokenizer
# import sys

# import simplejson as json
import json
import codecs


# from keras.models import load_model

# load model
# product_model = load_model("model_product_7.h5")
# intention_model = load_model("model_intention_7.h5")

# load dicts
word_dict = {}
category_dict = {}
intention_dict = {}

with codecs.open('word_dict.json', 'r', encoding='utf8') as f:
    word_dict = json.loads(f.read())
print(word_dict)
print(category_dict)
print(intention_dict)
