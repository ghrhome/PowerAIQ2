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
                speaker_a.append(l.strip()[3:])
            elif not l:
                None
            else:
                if a_speaking:
                    speaker_a.append(l)
                else:
                    speaker_b.append(l)
        f.close()
        id_list.append(txt_id)
        if len(speaker_a) == 0:
            if len(speaker_b) > 0:
                sp = speaker_b[0].split("A")
                if len(sp) > 1:
                    speaker_a.append(sp[1].strip())
        if len(speaker_b) == 0:
            if len(speaker_a) > 0:
                sp = speaker_a[0].split("B")
                if len(sp) > 1:
                    speaker_b.append(sp[1].strip())
        # still zero
        if len(speaker_a) == 0:
            print("Warning a zero")
            speaker_a.append("failed")
        if len(speaker_b) == 0:
            speaker_b.append("failed")
        train_items_a.append(convert_speaker_list_to_seg_id_list(speaker_a, word_dict))
        train_items_b.append(convert_speaker_list_to_seg_id_list(speaker_b, word_dict))

    return np.array(id_list), np.array(train_items_a), np.array(train_items_b)


def convert_speaker_list_to_seg_id_list(speaker_list, word_dict):
    seg_id_list = []
    for line in speaker_list:
        seg_list = jieba.cut(line)
        for seg in seg_list:
            if seg in word_dict:
                seg_id_list.append(word_dict[seg]["id"])
            else:
                print("Warning. Word not found.")
    return seg_id_list
