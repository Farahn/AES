import pandas as pd
import numpy as np
import csv
import json
import os
from io import StringIO
import io
import unicodedata
import re

#create a data directory if it does not exist
directory = 'data'
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory = 'data/ASAP'
if not os.path.exists(directory):
    os.makedirs(directory)

#preprocessing dine via the script available at https://github.com/nusnlp/nea/tree/master/data
#this script creates 5 fold CV data from the ASAP dataset (training_set_rel3.tsv) based on the essay IDs
path = os.path.join(os.getcwd(),'nea/data')
folders = ['fold_0', 'fold_1','fold_2','fold_3', 'fold_4']

#Specify which ASAP sets to use from 1-8
sets = [1,2]

def clean(t_):
    t_ = re.sub('\s+',' ',t_)
    t_ = re.sub('- ','',t_)
    #url_reg  = r'[a-z]*[:.]+\S+'
    #t_ = re.sub(url_reg, '', t_)
    t_ = re.sub('([.,!?()])', r' \1 ', t_)
    t_ = re.sub('\"', ' \" ',t_)
    t_ = re.sub('$', ' $ ',t_)
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

def clean_par(t_):
    t_ = re.sub('- ','',t_)
    t_ = re.sub('([.,!?()])', r' \1 ', t_)
    t_ = re.sub('\"', ' \" ',t_)
    t_ = re.sub('$', ' $ ',t_)
    t_ = re.sub(r'\'s', ' \'s', t_)
    t_ = re.sub(r'\'re', ' \'re', t_)
    t_ = re.sub(r'\'ll', ' \'ll', t_)
    t_ = re.sub(r'\'m', ' \'m', t_)
    t_ = re.sub(r'\'d', ' \'d', t_)
    t_ = re.sub(r'can\'t', 'can n\'t', t_)
    t_ = re.sub(r'n\'t', ' n\'t', t_)
    t_ = re.sub(r'sn\'t', 's n\'t', t_)
    #t_ = re.sub('\s{2,}', ' ', t_)
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

def clean_text(df, col = 'essay'):
    t = []
    t_par = []
    for i in df[col]:
        t.append(clean(i))
        t_par.append(clean_par(i))
    df['text1'] = t
    df['text_par'] = t_par
    df['label'] = df['domain1_score']
    return df


def data(path, sets, path_o, folders = ['fold_0', 'fold_1','fold_2','fold_3', 'fold_4']):
    
    for i in range(len(folders)):
        f_p = os.path.join(path,folders[i])
        f_path_o = os.path.join(path_o,folders[i])
        
        test_str = os.path.join(f_p,'test.tsv')
        train_str = os.path.join(f_p,'train.tsv')
        dev_str = os.path.join(f_p,'dev.tsv')

        df_test = pd.read_csv(test_str, sep = '\t')
        df_train = pd.read_csv(train_str, sep = '\t')
        df_dev = pd.read_csv(dev_str, sep = '\t')

        df_test = clean_text(df_test)
        df_train = clean_text(df_train)
        df_dev = clean_text(df_dev)

        for i in sets:
            path_i = os.path.join(f_path_o,str(i))
            if not os.path.exists(f_path_o):
                os.makedirs(f_path_o)

            if not os.path.exists(path_i):
                os.makedirs(path_i)
                
            df_test[df_test['essay_set']==i].to_csv(os.path.join(path_i,'test.csv'))
            df_train[df_train['essay_set']==i].to_csv(os.path.join(path_i,'train.csv'))
            df_dev[df_dev['essay_set']==i].to_csv(os.path.join(path_i,'dev.csv'))


data(path,sets,directory)

