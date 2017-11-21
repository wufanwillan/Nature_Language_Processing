#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import os
import porter
import re
from collections import Counter
import pandas as pd
import numpy as np
import random
import config
import shutil
import copy
import SVM
from MLP import MLP
import pickle as pk
import sys

eps=1e-24

class Test(object):
    def __init__(self,path="./tcdev_list",models="./model",dev=True,default_list="./stopword-list",outpath="./test-class-list"):
        self._path=path
        self._models=models+".pkl"
        self._dev=dev
        with open(default_list,'r') as f:
            self._stopword=f.read().split()
        self._frequency = pd.DataFrame([])
        self._data=pd.DataFrame([])
        self._count=0
        self._modelpath=models
        self._models=None
        self._outpath=outpath
        self._classlist=[]
        self._badfile=[]
        self._index = 0
        self._wordappear=config.testwordappear
        self._wordkeep=[]


    def Porter_extraction(self, file_path):
        p = porter.PorterStemmer()
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        r_num = '[0-9]'
        r = re.compile(r)
        r_num = re.compile(r_num)
        text = ''
        with open(file_path, 'r') as infile:
            output = ''
            word = ''
            line = infile.read()
            for c in line:
                if c.isalpha():
                    word += c.lower()
                else:
                    if word:
                        output += p.stem(word, 0, len(word) - 1)
                        word = ''
                    output += c.lower()
            text += (' ' + re.sub(r, ' ', re.sub(r_num, ' ', output)))
        return text

    def get_test(self):
        index=0
        with open(self._path,"r+") as f:
            while True:
                file=f.readline()
                if file:
                    if self._dev:
                        file_path,classname=file.split()
                        self._classlist.append(classname)
                    else:
                        file_path= file.split()
                    try:
                        sample_text = self.Porter_extraction(file_path)
                        sample_words = sample_text.split()
                        words_counter = Counter(sample_words)
                        words_keep = {key: val for key, val in words_counter.items() if val > self._wordappear}
                        # words_keep = {key: val for key, val in words_counter.items() if val > 0}
                        self._wordkeep.append(words_keep)
                    except:
                        self._badfile.append(self._index)
                    finally:
                        self._index+=1
                else:
                    break
        self._frequency=pd.DataFrame(self._wordkeep)
        return

    def load_weight(self):
        with open(self._modelpath,"rb") as f:
            self._models=pk.load(f)
        return

    def get_output(self):
        # frequency=self._frequency.drop(self._models[0][2],axis=1)
        # fre_sum=frequency.sum(axis=1).values
        # samples_array=frequency.fillna(0).values
        # samples_len=len(samples_array)
        # temp=samples_array/(fre_sum.reshape(samples_len,1)+eps)
        output_temp=[]
        output=[]
        index=set(self._badfile)
        bias=0
        output=[]
        output_array=[]
        classname=[]
        for model in self._models:
            classname.append(model[1])
            output_temp=[]
            drop_col=set(self._frequency.columns).difference(set(model[2]))
            temp=self._frequency.drop(drop_col,axis=1)
            keep_pd=pd.DataFrame([],columns=set(model[2]))
            data=pd.concat([keep_pd,temp])
            data_temp=data.drop("class",axis=1).fillna(0)
            data_array=data_temp.values
            data_sum=data_temp.sum(axis=1)
            samples_array=data_array/(data_sum.values.reshape(len(data_sum),1)+eps)
            for i in range(self._index):
                if i in index:
                    output_temp.append(1)
                    bias+=1
                else:
                    input_test=samples_array[i-bias]
                    input_test=np.ndarray.tolist(input_test)
                    output_temp.extend(model[0].propagate_forward(input_test))
                    # output_array=np.array(output_temp)
                    # output.append(self._models[np.argmax(output_array)][1])
            output_array.append(output_temp)
        out=np.array(output_array).T
        arg=np.argmax(out,axis=1)
        output=[classname[i] for i in arg]




        with open(self._outpath,"w") as f:
            with open(self._path,"r") as infp:
                count = 0
                while True:
                    inpath=infp.readline()
                    if inpath:
                        istream=inpath.split()[0]
                        f.write(istream+"\t"+output[count]+"\n")
                        count+=1
                    else:
                        break
        if self._dev:
            acc=0
            for i in range(self._index):
                if output[i]==self._classlist[i]:
                    acc+=1
            acc=acc/len(output)
            print(acc)
            return

if __name__ == '__main__':

    # test=Test(path="./tcdev_list",models="./model.pkl",dev=True,default_list="./stopword-list",outpath="./test-class-list")
    stopword_list, model, test_list, test_class_list = sys.argv[1:]
    test = Test(path=test_list, models=model, dev=True, default_list=stopword_list,
                outpath=test_class_list)
    # print("hidden: ",config.hidden)
    # print("epoch: ",config.epochs)
    # print("wordappear: ", config.wordappear)
    # print("testapper: ", config.testwordappear)
    test.get_test()
    test.load_weight()
    test.get_output()




