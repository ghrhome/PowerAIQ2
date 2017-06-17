from os import listdir
import json


def load_items_from_path(path):
    file_list = listdir(path)
    train_items = []
    for data_file in file_list:
        # get id based on file name
        txt_id = data_file.split(".")[0]
        # handle data file
        f = open(path + data_file)
        lines = f.readlines()
        f.close()
        speaker_a = []
        speaker_b = []
        for line in lines:
            if line.startswith("A"):
                speaker_a.append(line.strip()[2:])
            elif line.startswith("B"):
                speaker_b.append(line.strip()[2:])
        train_item = {"id": txt_id, "speaker_a": speaker_a, "speaker_b": speaker_b}
        train_items.append(train_item)
