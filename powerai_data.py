import json
import jieba
import numpy
import random

category_dict = {}
intention_dict = {}
word_dict = {}


def get_raw_tuple(file_name):
    with open('train_data.json') as data_file:
        train_data = json.load(data_file)

    speaker_a_list = []
    speaker_b_list = []
    category_list = []
    intention_list = []
    for item in train_data:
        # handle category dict
        category = item["category_result"]
        if category in category_dict:
            category_dict[category]["freq"] += 1
        else:
            category_dict[category] = {}
            category_dict[category]["id"] = len(category_dict)
            category_dict[category]["freq"] = 1
        category_list.append(category_dict[category]["id"])

        # handle category dict
        intention = item["intention_result"]
        if intention in intention_dict:
            intention_dict[intention]["freq"] += 1
        else:
            intention_dict[intention] = {}
            intention_dict[intention]["id"] = len(intention_dict)
            intention_dict[intention]["freq"] = 1
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
    (x_speaker_a_train, x_speaker_b_train, y_category_train, y_intention_train) = get_raw_tuple('train_data_run.json')
    (x_speaker_a_test, x_speaker_b_test, y_category_test, y_intention_test) = get_raw_tuple('test_data_run.json')
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
    s = json.dumps(category_dict, ensure_ascii=False, separators=(',', ':'))
    f = open('category_dict.json', 'wb')
    f.write(s.encode("utf-8"))
    f.close()
    s = json.dumps(intention_dict, ensure_ascii=False, separators=(',', ':'))
    f = open('intention_dict.json', 'wb')
    f.write(s.encode("utf-8"))
    f.close()
    s = json.dumps(word_dict, ensure_ascii=False, separators=(',', ':'))
    f = open('word_dict.json', 'wb')
    f.write(s.encode("utf-8"))
    f.close()


def generate_random_test_train_data(test_count=100):
    with open('train_data.json') as data_file:
        train_data = json.load(data_file)
    random.shuffle(train_data)

    y_data = train_data[:test_count]

    s = json.dumps(y_data, ensure_ascii=False, separators=(',', ':'))
    f = open('test_data_run.json', 'wb')
    f.write(s.encode("utf-8"))
    f.close()

    x_data = train_data[test_count:]

    s = json.dumps(x_data, ensure_ascii=False, separators=(',', ':'))
    f = open('train_data_run.json', 'wb')
    f.write(s.encode("utf-8"))
    f.close()


if __name__ == "__main__":
    print("Train-Test data is reset.")
    generate_random_test_train_data()
