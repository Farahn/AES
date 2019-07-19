import os
import os.path
import pandas as pd
from io import StringIO
import io
import unicodedata
import re

import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize

def read_data(raw_text):
    content = raw_text
    content = content.split() #splits the text by spaces (default split character)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def clean_(t_):
    t_ = re.sub('\s+',' ',t_)
    t_ = re.sub('- ','',t_)
    url_reg  = r'[a-z]*[:.]+\S+'
    t_ = re.sub(url_reg, '', t_)
    t_ = re.sub('([.,!?()])', r' \1 ', t_)
    t_ = re.sub(r'\'s', ' \'s', t_)
    t_ = re.sub(r'\'re', ' \'re', t_)
    t_ = re.sub(r'\'ll', ' \'ll', t_)
    t_ = re.sub(r'\'m', ' \'m', t_)
    t_ = re.sub(r'\'d', ' \'d', t_)
    t_ = re.sub(r'can\'t', 'can n\'t', t_)
    t_ = re.sub(r'n\'t', ' n\'t', t_)
    t_ = re.sub(r'sn\'t', 's n\'t', t_)
    t_ = re.sub('\s{2,}', ' ', t_)
    t_ = t_.lower()
    mydict = us_gb_dict()
    t_ = replace_all(t_, mydict)
    return(t_)

def us_gb_dict():    
    filepath = 'us_gb.txt'
    with open(filepath, 'r') as fp:  
        read = fp.read()
    us = []
    gb = []
    gb_f = True

    for i in read.splitlines():
        line = i.strip()
        #print(line)
        if line == "US":
            gb_f = False      
        elif gb_f == True:
            gb.append(line)
        else:
            us.append(line)
    us2gb = dict(zip(gb, us))
    return us2gb


def replace_all(text, mydict):
    
    for gb, us in mydict.items():
        text = text.replace(gb, us)
    return text


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X], dtype = int)

def zero_pad_test(X, seq_len_div):
    diff = seq_len_div - (len(X)%seq_len_div)
    return np.concatenate((np.array([x for x in X],dtype = int),np.zeros((diff,len(X[0])),dtype = int)), axis = 0)

def read_test_set(df_test, dictionary, SEQUENCE_LEN_D = 40, SEQUENCE_LEN = 65, BATCH_SIZE = 10):
    X_train = []
  
    for i in df_test['text1']:
        i = sent_tokenize(i)
        X_train.append([dictionary['START_SENT']])
        for j in i[:SEQUENCE_LEN_D-2]:
            x = read_data(str(j).lower())
            data = []
            data.append(dictionary['START'])
            for word in x:
                if word in dictionary:
                    index = dictionary[word]
                    count_iv_train += 1

                else:
                    index = dictionary['UNK']
                    count_oov_train += 1
                data.append(index)
            data.append(dictionary['END'])
            X_train.append(data)
        X_train.append([dictionary['END_SENT']])
        for k in range(max(SEQUENCE_LEN_D -  (len(i)+2), 0)):
            X_train.append([0]) # pad token maps to 0


    print('len of test set: ', len(X_train)//BATCH_SIZE)

        
    #X_test_ = zero_pad(X_test_, SEQUENCE_LEN)
    #X_test_ = zero_pad_test(X_test_, BATCH_SIZE*SEQUENCE_LEN_D) 
  
    rank_val = df_test['label']
    target_val = np.array(rank_val)
    onehot_encoder = OneHotEncoder(sparse=False)
    
    integer_encoded = target_val.reshape(len(target_val), 1)
    y_test = onehot_encoder.fit_transform(integer_encoded)

    return X_train, y_test

