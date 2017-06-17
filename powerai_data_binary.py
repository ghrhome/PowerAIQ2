#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import jieba
import numpy
import random
import pickle

category_dict = {
    u"教育险": {
        "freq": 0,
        "id": 8
    },
    u"健康险": {
        "freq": 0,
        "id": 3
    },
    u"家庭财产保险": {
        "freq": 0,
        "id": 2
    },
    u"人寿保险": {
        "freq": 0,
        "id": 6
    },
    u"意外伤害险": {
        "freq": 0,
        "id": 10
    },
    u"股票": {
        "freq": 0,
        "id": 4
    },
    u"理财": {
        "freq": 0,
        "id": 11
    },
    u"贵金属": {
        "freq": 0,
        "id": 5
    },
    u"期货": {
        "freq": 0,
        "id": 1
    },
    u"车险": {
        "freq": 0,
        "id": 7
    },
    u"基金": {
        "freq": 0,
        "id": 9
    }
}

intention_dict = {
    u"肯定": {
        "freq": 0,
        "id": 1
    },
    u"否定": {
        "freq": 0,
        "id": 2
    },
    u"疑问": {
        "freq": 0,
        "id": 3
    }
}
word_dict = {}





def get_raw_tuple(file_name):

    train_data = pickle.load(open(file_name, "rb"))
    speaker_a_list = []
    speaker_b_list = []
    category_list = []
    intention_list = []
    for item in train_data:
        # handle category dict
        category = item["category_result"]
        if category in category_dict:
            category_dict[category]["freq"] += 1
            category_list.append(category_dict[category]["id"])

        # handle category dict
        intention = item["intention_result"]
        if intention in intention_dict:
            intention_dict[intention]["freq"] += 1
            intention_list.append(intention_dict[intention]["id"])

        # handle speaker list for A and B
        speaker_a_list.append(convert_speaker_list_to_seg_id_list(item["speaker_a"]))
        speaker_b_list.append(convert_speaker_list_to_seg_id_list(item["speaker_b"]))

    return speaker_a_list, speaker_b_list, category_list, intention_list


def convert_speaker_list_to_seg_id_list(speaker_list):
    seg_id_list = []
    for line in speaker_list:
        seg_list = jieba.cut(line)
        for seg in seg_list:
            if seg in word_dict:
                word_dict[seg]["freq"] += 1
            else:
                word_dict[seg] = {}
                word_dict[seg]["id"] = len(word_dict)
                word_dict[seg]["freq"] = 1
            seg_id_list.append(word_dict[seg]["id"])
    return seg_id_list


def load_data(include_dict=True):
    (x_speaker_a_train, x_speaker_b_train, y_category_train, y_intention_train) = get_raw_tuple('train_data_run.bin')
    (x_speaker_a_test, x_speaker_b_test, y_category_test, y_intention_test) = get_raw_tuple('test_data_run.bin')
    x_a_train = numpy.array(x_speaker_a_train)
    x_b_train = numpy.array(x_speaker_b_train)
    y_cate_train = numpy.array(y_category_train)
    y_inte_train = numpy.array(y_intention_train)
    x_a_test = numpy.array(x_speaker_a_test)
    x_b_test = numpy.array(x_speaker_b_test)
    y_cate_test = numpy.array(y_category_test)
    y_inte_test = numpy.array(y_intention_test)
    save_dicts()

    return (x_a_train, x_b_train, y_cate_train, y_inte_train), (x_a_test, x_b_test, y_cate_test, y_inte_test)


def save_dicts():
    pickle.dump(category_dict, open("category_dict.bin", "wb"))
    pickle.dump(intention_dict, open("intention_dict.bin", "wb"))
    pickle.dump(word_dict, open("word_dict.bin", "wb"))


def generate_random_test_train_data(test_count=100):
    train_data = pickle.load(open('train_data.bin', "rb"))
    random.shuffle(train_data)
    y_data = train_data[:test_count]
    pickle.dump(y_data, open("test_data_run.bin", "wb"))
    x_data = train_data[test_count:]
    pickle.dump(x_data, open("train_data_run.bin", "wb"))


if __name__ == "__main__":
    print("Train-Test data is reset.")
    generate_random_test_train_data()
