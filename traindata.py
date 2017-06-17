from os import listdir
import json

label_dir_path = "./trainval/label/"
data_dir_path = "./trainval/data/"

data_files = listdir(data_dir_path)
label_files = listdir(label_dir_path)

train_items = []
for data_file in data_files:
    # get id based on file name
    txt_id = data_files[0].split(".")[0]
    # handle data file
    f = open(data_dir_path + data_file)
    lines = f.readlines()
    speaker_a = []
    speaker_b = []
    for line in lines:
        if line.startswith("A"):
            speaker_a.append(line.strip()[2:])
        elif line.startswith("B"):
            speaker_b.append(line.strip()[2:])
    f.close()
    # handle lable file
    label_file_name = label_dir_path + txt_id + ".label.txt"
    f = open(label_file_name)
    result_line = f.readline()
    f.close()
    result_param = result_line.split(",")
    category_result = result_param[0]
    intention_result = result_param[1]
    train_item = {"id": txt_id, "speaker_a": speaker_a, "speaker_b": speaker_b, "category_result": category_result,
                  "intention_result": intention_result}
    train_items.append(train_item)

s = json.dumps(train_items, ensure_ascii=False, separators=(',', ':'))

f = open('train_data.txt', 'wb')
f.write(s.encode("utf-8"))
f.close()
