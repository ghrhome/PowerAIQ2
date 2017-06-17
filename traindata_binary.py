#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import listdir
import pickle


label_dir_path = "./trainval/label/"
data_dir_path = "./trainval/data/"

data_files = listdir(data_dir_path)
label_files = listdir(label_dir_path)

train_items = []
for data_file in data_files:
    # get id based on file name
    txt_id = data_file.split(".")[0]
    # handle data file
    f = open(data_dir_path + data_file, "rb")
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
    # handle lable file
    label_file_name = label_dir_path + txt_id + ".label.txt"
    f = open(label_file_name)
    result_line = f.readline().strip()
    f.close()
    result_param = result_line.split(",")
    category_result = result_param[0].strip()
    intention_result = result_param[1].strip()
    train_item = {"id": txt_id, "speaker_a": speaker_a, "speaker_b": speaker_b, "category_result": category_result,
                  "intention_result": intention_result}
    train_items.append(train_item)

pickle.dump(train_items, open("train_data.bin", "wb"))
