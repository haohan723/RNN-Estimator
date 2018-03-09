train_x, train_y = [],[]
seq_size = 250
for i in range(len(features)-seq_size+1):
    train_x.append(features[i:i+seq_size])
    train_y.append(labels[i:i+seq_size])
    
def model_birnn(features,seq_len,mode):
  #RNN
    rnn_fcells = [tf.nn.rnn_cell.LSTMCell(dim) for dim in [128,256]]
    rnn_bcells = [tf.nn.rnn_cell.LSTMCell(dim) for dim in [128,256]]
    
    multi_rnn_fcell = tf.nn.rnn_cell.MultiRNNCell(rnn_fcells)
    multi_rnn_bcell = tf.nn.rnn_cell.MultiRNNCell(rnn_bcells)

    outputs,_,_ = tf.nn.bidirectional_dynamic_rnn(
                                   cell_fw = multi_rnn_fcell,
                                   cell_bw = multi_rnn_bcell,
                                   inputs = features,
                                   sequence_length = seq_len,
                                   dtype = tf.float32)
    outputs = tf.concat(outputs, axis=2)
    dense1 = tf.layers.dense(inputs = outputs,
                            units = 1024, 
                            activation = tf.nn.relu,
                            name = 'dense1')
    dense2 = tf.layers.dense(inputs = dense1,
                            units = 512, 
                            activation = tf.nn.relu,
                            name = 'dense1')
    dropout = tf.layers.dropout(inputs = dense2, rate =0.25,training=mode == tf.estimator.ModeKeys.TRAIN),
    logits = tf.layers.dense(inputs = dropout,
                            units = 10,
                            name = 'logits')
    return logits

model_birnn(train_x,seq_len,tf.estimator.ModeKeys.TRAIN)