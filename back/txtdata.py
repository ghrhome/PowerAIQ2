#-*- encoding:utf-8 -*-
from __future__ import absolute_import
from six.moves import zip
import numpy as np
import json
import warnings
import pickle
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence

def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)
    return np.array(new_sentences)


def load_data(path=u'评价语料_分词后.txt', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
              
    # Legacy support
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    #f1 = codecs.open(path, "r", encoding="utf-8")
    f1 = open(path, 'r') 
    print u"已经打开文本"

    # 获得句子列表，其中每个句子又是词汇的列表
    sentences_list = []
    for line in f1:
        single_sen_list = line.strip().split(" ")
        while "" in single_sen_list:
            single_sen_list.remove("")
        sentences_list.append(single_sen_list)
    print u"句子总数：", len(sentences_list)
    f1.close()

    # 读取语料类别标签
    f2 = open(u"语料类别.txt", 'r') 
    print u"已经打开文本"

    # 转为列表
    line_list = []
    for line in f2:
        line_list.append(line.strip())
    print u"列表里的元素个数：", len(line_list)

    f2.close()

    # 读取大语料文本
    f = open(u"dict.pkl", 'rb')  # 预先训练好的
    index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
    word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}
    new_dic = index_dict

    print u"Setting up Arrays for Keras Embedding Layer..."
    n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）

    # 划分训练集和测试集，此时都是list列表
    X_train_l, X_test_l, labels_train, labels_test = train_test_split(sentences_list, line_list, test_size=0.2)

    # 转为数字索引形式
    x_train = text_to_index_array(new_dic, X_train_l)
    x_test = text_to_index_array(new_dic, X_test_l)

    y_train = np.array(labels_train)  # 转numpy数组
    y_test = np.array(labels_test)

    # 将句子截取相同的长度maxlen，不够的补0
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return (x_train, y_train), (x_test, y_test)