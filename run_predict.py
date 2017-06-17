#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import keras
import powerai_data_binary as powerai_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import normalize
import sys
from keras.models import load_model
import pickle
import run_predict_data
import numpy2JSON as rr

max_words_a = 1024
max_words_b = 512

batch_size = 10
# load model
product_model = load_model("model_product_7.h5")
intention_model = load_model("model_intention_7.h5")

# load dicts
word_dict = pickle.load(open('word_dict.bin', "rb"))
category_dict = pickle.load(open('category_dict.bin', "rb"))
intention_dict = pickle.load(open('intention_dict.bin', "rb"))

category_map = []
for i in xrange(0, len(category_dict)):
    category_map.append("")
intention_map = []
for i in xrange(0, len(intention_dict)):
    intention_map.append("")

for key, value in category_dict.iteritems():
    category_map[value["id"]] = key

for key, value in intention_dict.iteritems():
    intention_map[value["id"]] = key

(id_list, test_a, test_b) = run_predict_data.load_data("./trainval/data/", word_dict)

print('Vectorizing sequence data...')
tokenizer_a = Tokenizer(num_words=max_words_a)
tokenizer_b = Tokenizer(num_words=max_words_b)
test_a = tokenizer_a.sequences_to_matrix(test_a, mode='binary')
test_b = tokenizer_b.sequences_to_matrix(test_b, mode='binary')

print(id_list)
print(test_a)
print(test_b)

for a in test_a:
    if len(a) < 2:
        print("!!!!!!!!!")
result_a = product_model.predict(test_a, batch_size=batch_size, verbose=1)

result_b = intention_model.predict(test_b, batch_size=batch_size, verbose=1)

# result_a = np.fromiter([np.argmax(r) for r in result_a], int)
# result_b = np.fromiter([np.argmax(r) for r in result_b], int)

print(result_a)
print(result_b)
# category_map = [u"期货", u"家庭财产保险", u"健康险", u"股票", u"贵金属", u"人寿保险", u"车险", u"教育险", u"基金", u"意外伤害险", u"理财"]
category_map = [u"家庭财产保险", u"期货", u"股票", u"健康险", u"理财", u"贵金属", u"车险", u"基金", u"教育险", u"意外伤害险", u"人寿保险"]
intention_map = [u"否定", u"肯定", u"疑问"]

rr.export_result(id_list, result_a, category_map, "production.json")
rr.export_result(id_list, result_b, intention_map, "intention.json")
