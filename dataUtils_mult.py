from __future__ import print_function, division

import os
import os.path
import pandas as pd
from io import StringIO
import io
import unicodedata
import re

import tensorflow as tf
import numpy as np
import collections
import random


from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize


def load_emb_glove(fp):
    glove_vocab = []
    #glove_embd=[]
    embedding_dict = {}

    file = open(fp,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
    file.close()

    print('Loaded GLOVE')
    return glove_vocab, embedding_dict, embed_vector

def read_data(raw_text):
    content = raw_text
    content = content.split() #splits the text by spaces (default split character)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dictionaries(words, n_max):
    count = collections.Counter(words).most_common(n_max-6) #creates list of word/count pairs;
    dictionary = {}
    c = 0
    dictionary['PAD'] = c #make the padding token 0, used to find sequence length
    c+=1
    dictionary['START_SENT'] = c #make the padding token 0, used to find sequence length
    c+=1
    dictionary['END_SENT'] = c #make the padding token 0, used to find sequence length
    c+=1
    dictionary['START'] = c
    c+=1
    dictionary['END'] = c
    c+=1
    dictionary['UNK'] = c
    c+=1
    for word, _ in count:
        dictionary[word] = c
        c+=1
    reverse_dictionary = {dictionary[k] : k for k in dictionary}
    return dictionary, reverse_dictionary

def vocab_dict(fpath, fpath_s, fpath_s1, glove_vocab, embedding_dict, embedding_dim, dname = 'nli_han', max_vocab = 50000):   

    #read training data and create vocab and embedding dictionaries
    
    directory = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    df_train = pd.read_csv(os.path.join(fpath, 'train.csv'))
    df_train_s = pd.read_csv(os.path.join(fpath_s, 'train.csv'))
    df_train_s1 = pd.read_csv(os.path.join(fpath_s, 'train.csv'))
    text = df_train['text1']
    df_val = pd.read_csv(os.path.join(fpath,'test.csv'))
    
    text_val = df_val['text1']
    text_train = df_train['text1']
    
    all_ = pd.concat([df_val, df_train])
    text_all_ = [all_['text1'],df_train_s['sent0'], df_train_s['sent1'],df_train_s1['sent0'], df_train_s1['sent1']]
    text_all = ''
    for i in range(len(text_all_)):
        text_all += ' '.join(str(elem) for elem in text_all_[i])
    
    training_data = read_data(text_all)

    #Create dictionary and reverse dictionary with word ids
    dictionary, reverse_dictionary = build_dictionaries(training_data, max_vocab)
    
    #save embedding dict
    import csv 
    dict_name = 'data/dict_' + dname + '.csv'    
    w = csv.writer(open(dict_name,'w'))
    for key,val in dictionary.items():
        w.writerow([key,val])
    
    dict_name = 'data/rev_dict_' + dname + '.csv'    
    w = csv.writer(open(dict_name,'w'))
    for key,val in reverse_dictionary.items():
        w.writerow([key,val])


    #Create embedding array
    doc_vocab_size = len(dictionary) 
    embeddings_tmp=[]
    c_g = 0
    c = 0


    for key in reverse_dictionary:
        item = reverse_dictionary[key]
        if item in glove_vocab:
            embeddings_tmp.append(embedding_dict[item])
            c_g = c_g + 1
        else:
            rand_num = np.random.uniform(low=-0.1, high=0.1,size=embedding_dim)
            embeddings_tmp.append(rand_num)
            c = c + 1

    # final embedding array corresponds to dictionary of words in the document
    embedding = np.asarray(embeddings_tmp)
    print('Vocabulary size:' , embedding.shape[0]) 
    print('Pre-trained Embeddings:', c_g)

    return dictionary, doc_vocab_size, embedding

def read_test_train_(fpath, fpath_s, dictionary, embedding, SEQUENCE_LEN = 50,
                    SEQUENCE_LEN_D = 25, tr = 0.85):   

    #read training data
    #three class labels; low, med, high
    #df = pd.read_csv('current_data/train_set_all.csv')
 
    df_train = pd.read_csv(os.path.join(fpath, 'train.csv'))
    df_train_s = pd.read_csv(os.path.join(fpath_s, 'train.csv'))
    #df_train_s = df_train_s[df_train_s['label']!=4]
    df_train = df_train.sample(frac=1.0)
    df_train = df_train.sample(frac = 1.0)
    text = df_train['text1']
    df_val = pd.read_csv(os.path.join(fpath,'test.csv'))
    
    text_val = df_val['text1']
    text_train = df_train['text1']
    rank_val = df_val['label']
    rank_train = df_train['label']
    rank_train_s = df_train_s['label']

    target_val = np.array(rank_val)
    target_train = np.array(rank_train)
    target_train_s = np.array(rank_train_s)

    onehot_encoder = OneHotEncoder(sparse=False)
    
    integer_encoded = target_train.reshape(len(target_train), 1)
    y_train = onehot_encoder.fit_transform(integer_encoded)
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(target_train_s)
    target_train_s = le.transform(target_train_s)
    print(list(le.inverse_transform([i for i in range(max(target_train_s)+1)])))
    integer_encoded_s = target_train_s.reshape(len(target_train_s), 1)
    y_train_s = onehot_encoder.fit_transform(integer_encoded_s)


    integer_encoded_val = target_val.reshape(len(target_val), 1)
    y_val = onehot_encoder.fit_transform(integer_encoded_val)
    
    #train set 
    count_oov_train = 0
    count_iv_train = 0
    X_train = []

    for i in df_train['text1']:
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
 

    print('OOV in training set: ', count_oov_train)
    print('IV in training set: ', count_iv_train)


 
    #val set
    X_val = []

    count_oov_test = 0
    count_iv_test = 0

    for i in df_val['text1']:
        i = sent_tokenize(i)
        X_val.append([dictionary['START_SENT']])        
        for j in i[:SEQUENCE_LEN_D-2]:
            x = read_data(str(j).lower())
            data = []
            data.append(dictionary['START'])
            for word in x:
                if word in dictionary:
                    index = dictionary[word]
                    count_iv_test += 1

                else:
                    index = dictionary['UNK']
                    count_oov_test += 1

                data.append(index)
            data.append(dictionary['END'])
            X_val.append(data)
        X_val.append([dictionary['END_SENT']])
        for k in range(max(SEQUENCE_LEN_D - (len(i)+2), 0)):
            X_val.append([0])

    print('OOV in test set: ', count_oov_test)
    print('IV in test set: ', count_iv_test)

    tr_len = int(tr*len(y_train))
    print(tr_len)

    X_train_s = []

    for i in range(len(df_train_s)):
        k = [(df_train_s.iloc[i]['sent0'])]
        k.append([(df_train_s.iloc[i]['sent1'])])
        for j in k:
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
            X_train_s.append(data)
        

    print('OOV in training set: ', count_oov_train)
    print('IV in training set: ', count_iv_train)

    tr_len_s = int(tr*len(y_train_s))


    return X_train[:tr_len*SEQUENCE_LEN_D], y_train[:tr_len], X_train[tr_len*SEQUENCE_LEN_D:], y_train[tr_len:], \
           X_train_s[:tr_len_s*2], y_train_s[:tr_len_s], X_train_s[tr_len_s*2:], y_train_s[tr_len_s:],X_val, y_val
