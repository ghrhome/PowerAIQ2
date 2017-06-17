#!usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

def walkpath(path):

    for dirpath, dirs, files in os.walk(path):  # 递归遍历当前目录和所有子目录的文件和目录
        print(files)
        for name in files:  # files保存的是所有的文件名
            if os.path.splitext(name)[1] == '.txt':
                filePath=dirpath+name
                readfile(filePath,name)

def readfile(filepath,name):
    file_input=open(filepath,"r")
    file_path="./output_a_b/"

    file_name_pre=name.split(".")[0]

    file_output_a=open(file_path+file_name_pre+"_a_label.txt","w")
    file_output_b=open(file_path+file_name_pre+"_b_label.txt","w")

    str_a=""
    str_b=""
    for line in file_input.readlines():
        if(line.startswith("A")):
            output_str=re.sub(r"A(：|:)","",line)
            str_a+=output_str+"\n"

        elif line.startswith("B"):
            output_str = re.sub(r"B(：|:)","",line)
            str_b+=output_str+"\n"

    file_output_a.write(str_a)
    file_output_b.write(str_b)
    file_output_a.close()
    file_output_b.close()

if __name__=="__main__":
    walkpath("./trainval/data/")