#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
import pickle
import jieba
import numpy as np


def load_data(dir_path, word_dict):
    data_files = listdir(dir_path)
    train_items_a = []
    train_items_b = []
    id_list = []
    for data_file in data_files:
        # get id based on file name
        txt_id = data_file.split(".")[0]
        # handle data file
        f = open(dir_path + data_file, "rb")
        lines = f.readlines()

        speaker_a = []
        speaker_b = []
        a_speaking = True
        for line in lines:
            l = line.strip()
            if l.startswith("A") or l.startswith(u"Ａ".encode("utf-8")):
                a_speaking = True
                speaker_a.append(l.strip()[2:])
            elif l.startswith("B") or l.startswith(u"Ｂ".encode("utf-8")):
                a_speaking = False
                speaker_b.append(l.strip()[2:])
            elif l.startswith(u"客户：".encode("utf-8")):
                a_speaking = False
                speaker_b.append(l.strip()[3:])
            elif l.startswith(u"客服：".encode("utf-8")):
                a_speaking = True
                speaker_b.append(l.strip()[3:])
            elif not l:
                None
            else:
                if a_speaking:
                    speaker_a.append(l)
                else:
                    speaker_a.append(l)
        f.close()
        id_list.append(txt_id)
        train_items_a.append(convert_speaker_list_to_seg_id_list(speaker_a, word_dict))
        train_items_b.append(convert_speaker_list_to_seg_id_list(speaker_b, word_dict))

    return np.array(id_list), np.array(train_items_a), np.array(train_items_b)


def convert_speaker_list_to_seg_id_list(speaker_list, word_dict):
    seg_id_list = []
    for line in speaker_list:
        seg_list = jieba.cut(line)
        for seg in seg_list:
            if word_dict[seg]:
                seg_id_list.append(word_dict[seg]["id"])
            else:
                print("Warning. Word not found." + seg)
    return seg_id_list
