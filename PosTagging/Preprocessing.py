#!/usr/bin/env python3
# _*_coding:utf-8_*_

# Preprocessing(filepath):
#
# task:preprocessing tag file into a counter dict{<T,W>:times} for Tag and Word\
#      and a list [Ti] for Tag transpose
#
# Input:relative file path or obsolute file path
#
# Output:dict{<T,W>:times} and transpose function P(Ti+1|Ti)

import os
from collections import Counter
import numpy as np
import pandas as pd
import re
import pickle

class Preprocessiming(object):
    def __init__(self,filepath):
        self.filepath=filepath
        self.fhandle='r'
        self.defaultline=1000
        self.tag_word_counter=Counter()
        self.tag_transpose_list=[]
        self.tag_transpose_set=set()
        self.tag_transpose_matrix=[]
        self.word_tag_matrix=[]
        self.pennTreeTags=["CC", "CD", "DT", "EX", "FW", "IN",
                            "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
                            "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
                            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
                            "$", "#", "``", "''", "-LRB-", "-RRB-", ",", ".", ":", "<s>",
                            "<|s>"]

    def Preprocessing(self):
        with open(self.filepath,self.fhandle) as f:
            linestriper=f.read().strip()
            # linespliter=re.split(r"[\n ]",linestriper)
            linespliter = re.split(r"[\n]", linestriper)
            for line in linespliter:
                line=' '.join([r"/<s>",line,r"/<|s>"])
                wordspliter=re.split(r" ",line)
                for word in wordspliter:
                    self.tag_word_counter[''.join(word.split("/")[:-1])]+=1
                    self.tag_transpose_set.add(word.split("/")[-1])
                    self.tag_transpose_list.append(word.split("/")[-1])
            self.tag_transpose_set.update([r"Unknown",r"<s>",r"<|s>"])
            tag_len=len(self.tag_transpose_set)
            word_len=len(self.tag_word_counter.keys())
            inital_probability=np.array([0]*pow(tag_len,2)).reshape(tag_len,tag_len)
            indexlist=list(self.tag_transpose_set)
            indexlist.sort()
            self.tag_transpose_matrix=pd.DataFrame(inital_probability,index=indexlist,columns=indexlist)
            tag_list_len=len(self.tag_transpose_list)
            for i in range(tag_list_len):
                if self.tag_transpose_list[i]=="<|s>":
                    continue
                else:
                    self.tag_transpose_matrix.ix[self.tag_transpose_list[i],self.tag_transpose_list[i+1]]+=1
            self.word_tag_matrix=pd.DataFrame(np.array([0]*(tag_len*word_len)).reshape(word_len,tag_len),index=self.tag_word_counter.keys(),columns=self.tag_transpose_set)
            for line in linespliter:
                line=' '.join([r"/<s>",line,r"/<|s>"])
                wordspliter=re.split(r" ",line)
                for word in wordspliter:
                    self.word_tag_matrix.loc[''.join(word.split("/")[:-1]),word.split("/")[-1]]+=1


    @property
    def Dict_checker(self):
        return self.tag_word_counter

    @property
    def Tag_checker(self):
        return self.tag_transpose_matrix

if __name__=='__main__':
    p=Preprocessiming('a2_data/sents.devt')
    p.Preprocessing()
    with open('tag_matrix.txt','wb') as f:
        pickle.dump(p.tag_transpose_matrix,f)
    with open('tag_word_counter.txt','wb') as f:
        pickle.dump(p.tag_word_counter,f)
    with open('word_tag_matrix.txt','wb') as f:
        pickle.dump(p.word_tag_matrix,f)
    with open('word_count.txt','wb') as f:
        pickle.dump(list(p.tag_word_counter),f)


