#!usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import numpy as np

npArray = np.array([[0.8, 0.12, 0.08], [0.56, 0.32, 0.12]])
resultMap = [u"肯定", u"否定", u"疑问"]


def walkpath(path):
    fileArr = []

    for dirpath, dirs, files in os.walk(path):  # 递归遍历当前目录和所有子目录的文件和目录
        for name in files:  # files保存的是所有的文件名
            if os.path.splitext(name)[1] == '.txt':
                fileArr.append(name)
    return fileArr


def npArr2List(npArr):
    return npArr.tolist()


def dataFormat(fileArr, resultsArr, resultMap):
    dataLen = len(fileArr)
    formatedArr = []
    resultMapLen = len(resultMap)
    for i in range(dataLen):
        result = resultsArr[i]

        for j in range(resultMapLen):
            tempObj = {
                "sampleId": fileArr[i],
                "prob": round(result[j], 4),
                "label": resultMap[j]
            }

            formatedArr.append(tempObj)

    submitData = {
        "type": "Fiance Product Classifcation",
        "result": formatedArr
    }

    return submitData


def toJSON(data):
    return json.dumps(data, ensure_ascii=False)


def save2File(data, name, path="."):
    file = open(path + "/" + name, "w")
    file.write(data.encode('utf-8'))
    file.close()


if __name__ == "__main__":
    arr = npArr2List(npArray)
    fileArr = walkpath("./trainval/data/")
    submitData = dataFormat(fileArr, arr, resultMap)
    save2File(toJSON(submitData))


def export_result(id_list, test_result, value_map, name):
    s = dataFormat(id_list, test_result, value_map)
    save2File(toJSON(s), name)
