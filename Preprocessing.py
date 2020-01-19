# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:28:06 2020

@author: Milad Ghanbari
"""


from collections import Counter
import numpy as np
import string
from nltk.corpus import stopwords


def TextProcess(dataset, NumFeat = 160): 
    allwords = []          
    all_words = []
    for i in range(len(dataset)):
        word = []
        word = dataset[i]['text']
        word = word.lower()
        word = word.split()
        for j in range(len(word)):
            if(word[j]!=''):
                all_words.append(word[j])
    
    for i in range(len(all_words)):
        all_words[i].encode('utf-8')
    allwords = ' '.join(all_words)
    final_vec = Counter(all_words)
    final_data = final_vec.most_common(NumFeat)
    list_word = []
    for i in range(len(final_data)):
        list_word.append(final_data[i][0])
    return list_word, final_data, allwords



def DatasetConstruct(dataset, list_word , feature_num = 164, With=True):
    l = len(dataset)
    X = np.zeros([l,feature_num])
    Y = np.zeros([l,1])
    X[ : , feature_num - 1] = 1

    stop_words= set(stopwords.words("english"))

    for i in range(l):
        punctuation_counter=0
        stopwords_counter=0    
        for j in range(len(list_word)):
            word_t = []
            word_t = dataset[i]['text']
            word_t = word_t.lower()
            word_t = word_t.split()
            X[i,j] = word_t.count(list_word[j])
            
            
        X[i,60] = dataset[i]['children']
        X[i,61] = dataset[i]['controversiality']
        X[i,62] = int(dataset[i]['is_root'])
        
        if(With):
            X[i,63] = X[i,60] * X[i,61]
            X[i,64] = X[i,60] * X[i,60]        
            for k in stop_words:
                stopwords_counter += dataset[i]['text'].count(k)
            if(stopwords_counter * (dataset[i]['children'])==0):
                X[i,65] = 0 
            else:
                X[i,65] = 1 / ((stopwords_counter) * (dataset[i]['children'])) 
                          
            for l in string.punctuation:
                punctuation_counter += dataset[i]['text'].count(l)
            if(punctuation_counter * (dataset[i]['children']) ==0):
                X[i,66] = 0 
            else:
                X[i,66] =  1/ ((punctuation_counter) * (dataset[i]['children']))
        
    
        Y[i] = dataset[i]['popularity_score']
    return X,Y
