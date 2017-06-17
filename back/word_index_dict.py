#-*- encoding:utf-8 -*-
import logging
import pickle

from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary

# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus("评价语料_分词后.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=100)  # 默认window=5
model.save("词向量.model")
model.wv.save_word2vec_format("词向量.txt", binary=False)

# 索引字典、词向量字典
index_dict, word_vectors= create_dictionaries(model)

# 存储为pkl文件
output = open("dict.pkl", 'wb')
pickle.dump(index_dict, output)  # 索引字典
pickle.dump(word_vectors, output)  # 词向量字典
output.close()
