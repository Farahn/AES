
import tensorflow as tf
import numpy as np

def cross_attention(inputs, seq_len_d, seq_len, batch_size, W_omega, time_major=False, return_alphas=False):
    """
    Attention mechanism layer for cross attention at sentence level; takes a sequence of RNN outputs 
    shaped [Batch_size*seq_len_d, sequence_len_sent, cell.output_size]
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """
   
    
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    #W_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
    #W_omega = tf.ones([hidden_size, hidden_size], tf.float32)
    
    #batch_size = inputs.shape[0]//seq_len_d
    batch_att_last = []
    batch_att_next = []
    output = []

    for i in range(batch_size):
        inp = inputs[i*seq_len_d:(i+1)*seq_len_d,:,:]
        seq_len_batch = tf.reshape(seq_len[i*seq_len_d:(i+1)*seq_len_d], [-1])
        pad_ = tf.expand_dims(tf.zeros_like(inp[0,:,:]),0)
        inp = tf.concat([pad_,inp,pad_], axis = 0)
        seq_len_batch = tf.concat([[0],seq_len_batch,[0]], axis = 0)
       
        for k in range(1,seq_len_d+1):
            inp_last = tf.expand_dims(inp[k-1,:,:],0)
            inp_next = tf.expand_dims(inp[k+1,:,:],0)
            inp_current = tf.transpose(inp[k,:,:])
            seq_len_last = seq_len_batch[k-1]
            #seq_len_current = seq_len_batch[k]
            seq_len_next = seq_len_batch[k+1]
            batch_att_last.append(attention_h(inp_last, W_omega, inp_current, seq_len_last))
            batch_att_next.append(attention_h(inp_next, W_omega, inp_current, seq_len_next))
          
    batch_next = tf.concat(batch_att_next, 0)
    batch_last = tf.concat(batch_att_last, 0)
    output = tf.concat([batch_next,batch_last,inputs], 2)
        #output.append(batch_att_last)
            
    return output
            
def attention_h(inputs, W_omega, u_omega, seq_len, return_alphas=False):
    
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # One fully connected layer for each of the hidden states;
    #  the shape of `v` is (B*T,D)
    v = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega)
    # For each of the B*T hidden states its vector of size A from `v` is reduced with `u` vector
    vu = tf.matmul(v, u_omega)   # (T,T) shape
    vu = tf.transpose(vu)        # (T,T) shape
    
    seq_len_t = tf.tile([seq_len],[inputs.shape[1].value])
    mask = tf.sequence_mask(seq_len_t , inputs.shape[1].value)
    attn_mask = tf.multiply(-30.0,tf.ones_like(vu))
    attn_mask_z = tf.zeros_like(vu)
    attn_pad = tf.where(mask,x = attn_mask_z,y = attn_mask)
    '''scores_exp = tf.exp(tf.add(vu,attn_pad))
    
    scores_sum = tf.reduce_sum(tf.exp(vu), axis=0)
    alphas = tf.truediv(scores_exp, scores_sum)
    '''
    vu = tf.add(vu,attn_pad)
    alphas = tf.nn.softmax(vu)                       # (B,T) shape also
    

    #inp_t = tf.stack([inputs for _ in range(alphas.shape[0].value)],0)
    inp_t = inputs
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reshape(tf.reduce_sum(inp_t * tf.expand_dims(alphas, -1), 1),tf.shape(inputs))
    
    #output = alphas
    return output



