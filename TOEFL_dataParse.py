
import pandas as pd
import numpy as np
import csv
import json
import os
from io import StringIO
import io
import unicodedata
import re


def get_text_data(path, cols, csv_path):
    w = csv.writer(open(csv_path,'w'))
    w.writerow(cols)
    l_files = os.listdir(path)
    for l in l_files:
        fpath = os.path.join(path, l)
        with open(fpath) as file:
            w.writerow([l, file.read()])


path = os.path.join(os.getcwd(),'ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original')

directory = 'data'
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory = 'data/TOEFL'
if not os.path.exists(directory):
    os.makedirs(directory)
    
data_path = os.path.join(directory, 'textdata.csv')
get_text_data(path, ['Filename', 'text'], data_path)
text_pd = pd.read_csv(data_path)

path_csv = os.path.join(os.getcwd(),'ETS_Corpus_of_Non-Native_Written_English/data/text')
l_csv = ['train', 'dev', 'test']

l_csvfiles = os.listdir(path_csv)
for i in l_csvfiles:
    if l_csv[0] in i:
        pd_ = pd.read_csv(os.path.join(path_csv,i), header = None, names = ['Filename', 'prompt', 'lang', 'score'])
        df_train = pd_.merge(text_pd ,on='Filename', how = 'inner')
    if l_csv[1] in i:
        pd_ = pd.read_csv(os.path.join(path_csv,i), header = None, names = ['Filename', 'prompt', 'lang', 'score'])
        df_dev = pd_.merge(text_pd ,on='Filename', how = 'inner')
    if l_csv[2] in i:
        pd_ = pd.read_csv(os.path.join(path_csv,i), header = None, names = ['Filename', 'prompt', 'lang', 'score'])
        df_test = pd_.merge(text_pd ,on='Filename', how = 'inner')


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

def one_hot_score(df, col = 'score'):
    s = []
    for i in df[col]:
        if str(i) == 'low':
            s.append(1)
        if str(i) == 'medium':
            s.append(2)
        if str(i) == 'high':
            s.append(3)
    
    df['label'] = s
    return df
    

def clean_text(df, col = 'text'):
    t = []
    t_par = []
    for i in df[col]:
        t.append(clean(i))
        t_par.append(clean_par(i))
    df['text1'] = t
    df['text_par'] = t_par
    return df


df_test['score'] = df_test['lang']


df_test = clean_text(df_test)
df_test = one_hot_score(df_test)
df_test.to_csv(os.path.join(directory,'test.csv'))


df_train = clean_text(df_train)
df_train = one_hot_score(df_train)
df_train.to_csv(os.path.join(directory,'train.csv'))


df_dev = clean_text(df_dev)
df_dev = one_hot_score(df_dev)
df_dev.to_csv(os.path.join(directory,'dev.csv'))


os.remove(data_path)

