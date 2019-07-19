

import tensorflow as tf
import numpy as np

def ordloss_m(y_hat, y_true, batch_size):
    """
    y_hat has dims num_classes-1; computes negative log likelihood
    """
    eps = np.array([1.0e-25])
   
    s_max = tf.sigmoid(y_hat)
    y_p = tf.concat([tf.constant(0.,shape = [batch_size,1]),s_max,tf.constant(1.,shape = [batch_size,1])],1)    
    
    y_t = tf.add(1, tf.cast(tf.argmax(y_true,1), dtype = tf.int32))
    
    y_t_1 = tf.cast(tf.argmax(y_true,1), dtype = tf.int32)
    
    cat_idx_1 = tf.stack([tf.range(0, tf.shape(y_p)[0]), y_t_1], axis=1)
    result_1 = tf.gather_nd(y_p, cat_idx_1)
    
    cat_idx = tf.stack([tf.range(0, tf.shape(y_p)[0]), y_t], axis=1)
    result = tf.gather_nd(y_p, cat_idx)
    r = tf.add(tf.maximum(tf.subtract(result,result_1),0.0),eps)
    
    loss = tf.reduce_mean(tf.negative(tf.log(r)))
    
    return loss

def preds(y_hat, batch_size):
    """
    y_hat has dims num_classes-1; computes most likely class
    """
    
    s_max = tf.sigmoid(y_hat)
    y_p = tf.concat([tf.constant(0.,shape = [batch_size,1]),s_max,tf.constant(1.,shape = [batch_size,1])],1)    
    
    r = tf.shape(y_p)[1]-1
    px = tf.subtract(y_p[:,1:],y_p[:,:r])
    
    preds = tf.argmax(px,axis = 1)
    return preds, px



def ordloss_mult(y_hat, y_true, y_true_l, batch_size):
    """
    y_hat has dims num_classes-1; computes negative log likelihood
    """
    eps = np.array([1.0e-25])
   
    s_max = tf.sigmoid(y_hat)
    y_p = tf.concat([tf.constant(0.,shape = [batch_size,1]),s_max,tf.constant(1.,shape = [batch_size,1])],1)    
    
    y_t = tf.add(1, tf.cast(tf.argmax(y_true,1), dtype = tf.int32))
    
    y_t_1 = tf.cast(y_true_l, dtype = tf.int32)
    
    cat_idx_1 = tf.stack([tf.range(0, tf.shape(y_p)[0]), y_t_1], axis=1)
    result_1 = tf.gather_nd(y_p, cat_idx_1)
    
    cat_idx = tf.stack([tf.range(0, tf.shape(y_p)[0]), y_t], axis=1)
    result = tf.gather_nd(y_p, cat_idx)
    r = tf.add(tf.maximum(tf.subtract(result,result_1),0.0),eps)
    
    loss = tf.reduce_mean(tf.negative(tf.log(r)))
    
    return loss

