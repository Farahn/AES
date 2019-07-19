import os
import os.path
import pandas as pd
from io import StringIO
import io

import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

SEQUENCE_LENGTH = 40
SEQUENCE_LENGTH_D = 25

#read train and val data
fpath = 'data/TOEFL'


SEQUENCE_LEN_D = SEQUENCE_LENGTH_D
SEQUENCE_LEN = SEQUENCE_LENGTH

df_train = pd.read_csv(os.path.join(fpath, 'train.csv'))
text = df_train['text1']
df_val = pd.read_csv(os.path.join(fpath,'test.csv'))

text_val = df_val['text1']
text_train = df_train['text1']
rank_val = df_val['label']
rank_train = df_train['label']

target_val = np.array(rank_val)
target_train = np.array(rank_train)

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = target_train.reshape(len(target_train), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

integer_encoded_val = target_val.reshape(len(target_val), 1)
y_test = onehot_encoder.fit_transform(integer_encoded_val)

### Get bert senence mebeddings for each sentence in an essay
## Based on BERT serving client https://pypi.org/project/bert-serving-client/
## base uncased model
## Pooling = NONE for token level representation
## bert-serving-start -model_dir /temp/uncased_L-12_H-768_A-12/ -num_worker=4 -pooling_strategy=NONE -max_seq_len=40
from nltk import sent_tokenize
from bert_serving.client import BertClient
bc = BertClient()
print('Starting BERT client..')
b_len = 768 
X_train = []

for i in df_train['text1']:
    i = sent_tokenize(i)
    X_train.extend(bc.encode(i[:SEQUENCE_LEN_D]))
    for k in range(max(SEQUENCE_LEN_D - (len(i)), 0)):
        X_train.append([[0]*b_len]*SEQUENCE_LEN) # pad token maps to 0
X_train = np.array(X_train)  

np.save('data/TOEFL/X_train_TOEFL', X_train)
np.save('data/TOEFL/y_train_TOEFL', y_train)


del X_train
del y_train
import gc
gc.collect()


#test set
X_test = []

for i in df_val['text1']:
    i = sent_tokenize(i)
    X_test.extend(bc.encode(i[:SEQUENCE_LEN_D]))
    for k in range(max(SEQUENCE_LEN_D - (len(i)), 0)):
        X_test.append([[0]*b_len]*SEQUENCE_LEN) # pad token maps to 0
X_test = np.array(X_test)


np.save('data/TOEFL/X_test_TOEFL', X_test)
np.save('data/TOEFL/y_test_TOEFL', y_test)

