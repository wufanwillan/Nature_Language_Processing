#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import Preprocessing
import pickle
import math
import numpy as np
import pandas as pd
import random
import re

class Train(object):
    def __init__(self,tag_matrix,tag_word,word_count=None,test_sentence=None,test_label=None):
        self.matrix=tag_matrix
        self.tag_word=tag_word
        self.test_sentence=test_sentence
        self.bias=0
        self.len_of_test=len(test_sentence)+1
        self.len_of_tag=len(tag_matrix)
        self.word_strip=test_sentence.strip()
        self.word_split=self.word_strip.split(' ')
        self.viterbi_matrix=pd.DataFrame(np.array([0]*(self.len_of_test*self.len_of_tag)).reshape(self.len_of_test,self.len_of_tag))
        self.word_count=word_count
        self.viterbi_back_monitor = pd.DataFrame(
            np.array([0] * (self.len_of_test * self.len_of_tag)).reshape(self.len_of_test, self.len_of_tag))
        self.path=[]
        self.label=test_label
        self.taglist=list(tag_matrix.index)

    def Add_one_smooth(self):
        # lenrow, lencol = len(self.matrix), len(self.matrix.iloc[0])
        self.smooth_matrix=self.matrix+1
        self.smooth_word_tag=self.tag_word+1
        self.bias=0

    def Generate_probability(self,tag_alg="Add_one_smooth"):
        tag_alg=getattr(self,tag_alg)
        total_matrix_sum=sum(self.smooth_matrix.apply(lambda x:x.sum(),axis=1))+self.bias
        total_word_tag_sum=sum(self.smooth_word_tag.apply(lambda x:x.sum(),axis=1))+self.bias
        lenrow, lencol = len(self.smooth_matrix), len(self.smooth_matrix.iloc[0])
        for i in range(lenrow):
            for j in range(lencol):
                self.smooth_matrix.iloc[i,j]=math.log(self.smooth_matrix.iloc[i,j]/total_matrix_sum)
        lenrow1, lencol1 = len(self.smooth_word_tag), len(self.smooth_word_tag.iloc[0])
        for i in range(lenrow1):
            for j in range(lencol1):
                self.smooth_word_tag.iloc[i,j]=math.log(self.smooth_word_tag.iloc[i,j]/total_word_tag_sum)

    def Viterbi_Layer(self,lastlayer):
        for i in range(self.len_of_tag):
            if self.word_split[i] in self.word_count:
                emission_sery=self.viterbi_matrix.iloc[lastlayer]+self.smooth_matrix.iloc[i]+self.smooth_word_tag.ix[self.word_split[i],i]
                self.viterbi_matrix.iloc[lastlayer+1,i]=max(emission_sery)
                self.viterbi_back_monitor.iloc[lastlayer+1,i]=emission_sery.index(max(emission_sery))
            else:
                self.viterbi_matrix.iloc[lastlayer + 1, i] = max(
                    self.viterbi_matrix.iloc[lastlayer] + self.smooth_matrix.iloc[i] + self.smooth_word_tag.ix[
                        self.word_split[i], "Unknown"])
            self.viterbi_back_monitor.iloc[lastlayer + 1, i] = emission_sery.index(max(emission_sery))

    def Viterbi_alg(self):
        for i in range(1,self.len_of_test):
            self.Viterbi_Layer(i)
        self.viterbi_matrix.iloc[-1]=self.viterbi_matrix.iloc[-2]+self.smooth_matrix.iloc[:,-1]

    def Cal_accuracy(self):
        correct=0
        total=0
        self.Generate_probability()
        self.Viterbi_alg()
        self.Back_pro()
        for i in range(self.len_of_test):
            if self.path[i]==self.label[i]:
                correct+=1
            total+=1
        return correct/sum

    def findmore(self,test):
        with open("output.test",'w') as f:
            lines=re.split(r"[\n ]",test)
            for line in lines:
                if line in self.tag_word.index:
                    tag_condidates=list(self.tag_word.loc[line])
                    tag_index=tag_condidates.index(max(tag_condidates))
                    tag=self.taglist[tag_index]
                    output=''.join([line,"/",tag," "])
                    f.write(output)



    def K_folder_validation(self,k=10):
        shuffle=random.shuffle(range(self.len_of_test))
        devide=self.len_of_test/k
        train_set=self.test_sentence

p=Preprocessing.Preprocessiming('a2_data/sents.devt')
p.Preprocessing()
input_sentence=open("a2_data/sents.test").read()
t=Train(p.tag_transpose_matrix,p.word_tag_matrix,list(p.tag_word_counter),input_sentence)
t.findmore(input_sentence)

