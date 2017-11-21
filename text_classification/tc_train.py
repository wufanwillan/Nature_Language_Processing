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

class Preprocess(object):

    def __init__(self,default_path="./train-class-list",default_list="./stopword-list",crossvalidation=False,threshold=3,hidden=1):

        self._train_file_path=default_path
        self._class_name=set()
        with open(default_list,'r') as f:
            self._stopword=f.read().split()
        #self._frequency=np.zeros((1,len(self._stopword)))
        # self._stopword.append("class")
        self._frequency=pd.DataFrame([])
        self._frequentword=pd.DataFrame([])
        self._count=0
        self._crossvalidation=crossvalidation
        self._posdata=None
        self._nagdata=None
        self._threshold=config.threshold
        self._hidden=hidden
        self._dropindex=set(self._stopword)
        self._wordappear=config.wordappear
        self._cachecolumns=None
        # self._keepindex={}
        # self._keepset=[]
        # self._counter=Counter()

    def Porter_extraction(self,file_path):
        p = porter.PorterStemmer()
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        r_num='[0-9]'
        r=re.compile(r)
        r_num=re.compile(r_num)
        text=''
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
            text+=(' '+re.sub(r,' ',re.sub(r_num,' ',output)))
        return text

    def Chisquare_test(self,classname):
        class_temp=self._frequentword[self._frequentword["class"]==classname]
        notin_temp=self._frequentword[self._frequentword["class"]!=classname]
        pos_len=len(class_temp)
        neg_len=len(notin_temp)
        # print(pos_len,neg_len)
        res=[]
        for i in range(len(class_temp.columns) - 1):
            feature = class_temp.columns[i]
            if feature!="class":
            # n00 = len(notin_temp.loc[notin_temp[feature] == 0])
            # n01 = len(class_temp.loc[class_temp[feature] == 0])
            # n10 = len(notin_temp.loc[notin_temp[feature] != 0])
            # n11 = len(class_temp.loc[class_temp[feature] != 0])
                n10 = notin_temp[feature].count()
                n11 = class_temp[feature].count()
                n00=neg_len-n10
                n01=pos_len-n11
                try:
                    chi = ((n00 + n01 + n10 + n11) * pow((n11 * n00 - n10 * n01), 2) )/ (
                    (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00))
                except:
                    chi=0
            else:
                chi=10
            res.append(chi)
        dropindex=[class_temp.columns[i] for i in range(len(class_temp.columns)-1) if res[i]<self._threshold]
        pos_temp=class_temp.drop(dropindex,axis=1)
        neg_temp=notin_temp.drop(dropindex,axis=1)
        self._posdata=pos_temp.fillna(0)
        self._nagdata=neg_temp.fillna(0)
        self._cachecolumns=set(pos_temp.columns)
        # print(self._cachecolumns,self._cachecolumns[:5])
        # print(len(pos_temp.columns))
        return

    def feature_Selection(self):
        return self._dropindex


    def get_trainclass(self):

        index=0
        text_index=[]
        text_word=[]
        text_freword=[]
        class_name=[]
        counter_cache=[self._frequency]
        keep_cache=[self._frequentword]
        if self._crossvalidation:
            self.Cross_validation()
        with open(self._train_file_path,"r") as f:
            print("preprocessing=======>"+self._train_file_path)
            temp_read=f.read()
            temp_file=temp_read.split("\n")
            for temp in temp_file[:2]:
                sample_path,sample_class=temp.split()
                self._class_name.add(sample_class)
                try:
                    sample_text=self.Porter_extraction(sample_path)
                    sample_words=sample_text.split()
                    words_counter=Counter(sample_words)
                    index+=1
                    words_keep={key:val for key,val in words_counter.items() if val>self._wordappear}
                    text_freword.append(words_keep)
                except:
                    pass
                words_keep_pd=pd.DataFrame(text_freword)
                keep_cache.append(words_keep_pd)
                text_index.append(sample_class)
                index=0
        self._frequentword = pd.concat(keep_cache, axis=0, ignore_index=True)
        self._frequentword["class"]=text_index
        dropset=set(self._stopword).intersection(set(self._frequentword.columns))
        self._frequentword.drop(dropset,axis=1,inplace=True)
        return

    def Cross_validation(self):
        dev_name=self._train_file_path+"dev_set"
        os.mkdir(dev_name)
        with open(self._train_file_path+"dev_list_class", "w+") as f:
            with open(self._train_file_path+"dev_list", "w+") as g:
                files = os.listdir(self._train_file_path)
                for path in files:
                    class_path = self._train_file_path + '/' + path
                    if os.path.isdir(class_path):
                        train_samples=os.listdir(class_path)
                        random.shuffle(train_samples)
                        dev_sample=train_samples[:(int(len(train_samples)*config.dev_ratio))]
                        for sample in dev_sample:
                            ori_path=class_path+"/"+sample
                            shutil.move(ori_path,dev_name)
                            full_path=dev_name+"/"+sample+"\t"+path+"\n"
                            f.write(full_path)
                            g.write(dev_name+"/"+sample+"\n")
                            # shutil.move(,""tcdev_set")
        return

    def fit_svm(self):
        from numpy import linalg
        def linear_kernel(x1, x2):
            return np.dot(x1, x2)

        def polynomial_kernel(x, y, p=3):
            return (1 + np.dot(x, y)) ** p

        def gaussian_kernel(x, y, sigma=5.0):
            return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

        pos_samples=self._posdata.drop("class",axis=1)
        pos_labels=self._posdata.loc[:,"class"]
        nag_samples = self._nagdata.drop("class", axis=1)
        nag_labels = self._nagdata.loc[:, "class"]
        pos_array=pos_samples.values
        pos_arraylabels= np.ones((len(pos_samples),1))
        nag_array = nag_samples.values
        nag_arraylabels = np.ones((len(nag_samples),1))*-1
        # clf=SVM.SVM(gaussian_kernel)
        # clf = SVM.SVM(C=1)
        clf = SVM.SVM()
        x_train=np.vstack((pos_array,nag_array))
        y_train=np.vstack((pos_arraylabels,nag_arraylabels))
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_train)
        correct = np.sum(y_predict == y_train)
        acc=correct/len(self._frequency)

    def f_step(self,o):
        return 1 if o>0 else -1

    def fit_mlp(self,classname,hidden=5,lrate=0.001,momentum=0.9):
        self.Chisquare_test(classname)
        def learn(network, samples, epochs=config.batchsize, lrate=0.01, momentum=0.1):
            # Train
            seq = [x for x in range(samples.size)]
            for k in range(config.epochs):
                network.count = []
                random.shuffle(seq)
                for n in seq:
                    # n = np.random.randint(samples.size)
                    network.propagate_forward(samples['input'][n])
                    network.propagate_backward(samples['output'][n], lrate, momentum)
                network.loss.append(sum(network.count) / len(seq))
            # print(network.loss)
            # Test
            acc=[]
            for i in range(samples.size):
                o = network.propagate_forward(samples['input'][i])
                diff=abs(self.f_step(o)-samples['output'][i])/2
                acc.append(diff)
            acc_ratio=1-sum(acc)/samples.size
            print("MLP"+classname+" dev_accracy=============>")
            print(acc_ratio)

        pos_samples=self._posdata.drop("class",axis=1)
        # pos_samples = pos_samples.drop(self._dropindex, axis=1)
        pos_sum=pos_samples.sum(axis=1)
        pos_ln=len(pos_sum)
        pos_labels=self._posdata.loc[:,"class"]

        nag_samples = self._nagdata.drop("class", axis=1)
        # nag_samples = nag_samples.drop(self._dropindex, axis=1)
        nag_sum=nag_samples.sum(axis=1)
        nag_ln=len(nag_sum)
        nag_labels = self._nagdata.loc[:, "class"]

        pos_array=pos_samples.values
        temp=pos_array/(pos_sum.values.reshape(pos_ln,1)+eps)
        pos_array=temp
        pos_arraylabels= np.ones((len(pos_samples),1))
        nag_array = nag_samples.values
        temp=nag_array/(nag_sum.values.reshape(nag_ln, 1)+eps)
        nag_array=temp
        nag_arraylabels = np.ones((len(nag_samples),1))*-1

        x_train=np.vstack((pos_array,nag_array))
        y_train=np.vstack((pos_arraylabels,nag_arraylabels))

        input=np.ndarray.tolist(x_train)
        output=np.ndarray.tolist(y_train)
        f_len=len(pos_samples.columns)
        d_len=len(x_train)
        if hidden>0:
            network=MLP(f_len,hidden,1)
        else:
            network = MLP(f_len, 1)
        samples = np.zeros(d_len, dtype=[('input', float, f_len), ('output', float, 1)])

        def f(i):
            samples[i]=x_train[i],y_train[i]

        [f(i) for i in range(d_len)]

        network.reset()
        learn(network, samples)

        # network.plot_loss()
        return network










if __name__ == "__main__":

    hold=5
    models=[]
    # default_path, default_list, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # p = Preprocess(default_list=default_list, default_path=default_path)
    p = Preprocess()
    # p.Cross_validation()
    # outpath=model_path
    p.get_trainclass()
    print("training...")
    for classname in p._class_name:
        network=p.fit_mlp(classname,hidden=config.hidden)
        class_name=classname
        feature=p._cachecolumns
        models.append([network,class_name,feature])
    with open(outpath,"wb") as f:
        pk.dump(models,f)


