from __future__ import print_function
import numpy as np
import random, itertools

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def zero_pad_test(X, seq_len_div):
    if (len(X)%seq_len_div) == 0:
        return np.array([x for x in X])
    diff = seq_len_div - (len(X)%seq_len_div)
    return np.concatenate((np.array([x for x in X]),np.zeros((diff,len(X[0])))), axis = 0)

def batch_generator(X, y, batch_size, seq_len = 1, shuffle = True):
    """Primitive batch generator 
    """
    size = X.shape[0]//seq_len
    X_copy = X.copy()
    y_copy = y.copy()

    if shuffle:
        # group X by seq_len
        grouped = list(zip(*[iter(X_copy)]*seq_len))
        z = list(zip(grouped, y_copy))
        random.shuffle(z)
        X_copy, y_copy = zip(*z)
        X_copy = np.array(list(itertools.chain.from_iterable(X_copy)))
        
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i * seq_len:(i + batch_size)* seq_len], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            if shuffle:
                grouped = list(zip(*[iter(X_copy)]*seq_len))
                z = list(zip(grouped, y_copy))
                random.shuffle(z)
                X_copy, y_copy = zip(*z)
                X_copy = list(itertools.chain.from_iterable(X_copy))
            continue

def batch_generator_par(X, y, X_par, seq_len = 1):
    """Primitive batch generator
    generates batches of size one:
    X: size seq_len_d * num pars in doc i
    """
    par = X_par.copy()
    par.insert(0,0)
    X_copy = X.copy()
    y_copy = y.copy()

    i = 0
    while True:
        if i < len(y):
            yield X_copy[sum(par[:i+1])*seq_len:sum(par[:i+2])*seq_len], np.reshape(y_copy[i],(1,-1)), par[i+1]
            i = i + 1
        else:
            i = 0
            continue


def test_batch_generator(X, batch_size, seq_len = 1):
    """Primitive batch generator 
    """
    size = X.shape[0]//seq_len
    X_copy = X.copy()

    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i * seq_len:(i + batch_size)* seq_len]
            i += batch_size
        else:
            i = 0
            continue
