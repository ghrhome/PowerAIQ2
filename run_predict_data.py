#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
import pickle
def load_data(dir_path):
    data_files = listdir(dir_path)


    train_items = []
    for data_file in data_files:
        # get id based on file name
        txt_id = data_file.split(".")[0]
        # handle data file
        f = open(data_dir_path + data_file, "rb")
        lines = f.readlines()
        f.close()
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