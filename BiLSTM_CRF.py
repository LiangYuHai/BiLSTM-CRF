import tensorflow as tf
import numpy as np
import matplotlib as plt

class Model:
    def init(self, config):
        self.LR=config['LR']
        self.BATCH_SIZE=config['BATCH_SIZE']
        self.EMBEDDING_DIM=config['EMBEDDING_DIM']
        self.SEN_LEN=config['SEN_LEN']
        self.EMBEDDING_SIZE=config['EMBEDDING_SIZE'] #还要+1? 空白to 0
        self.TAG_SIZE=config['TAG_SIZE']

        # config["pretrained"]=False
    def build_net(self):
        self.tf_x=tf.placeholder(dtype=tf.int32,shape=[None,self.SEN_LEN])
        self.tf_y=tf.placeholder(dtype=tf.int32,shape=[None,self.SEN_LEN])
        self.keep_prob=tf.placeholder(tf.float32)
        word_embedding=tf.get_variable('word_embedding',shape=[self.EMBEDDING_SIZE,self.EMBEDDING_DIM],dtype=tf.float32)
        input_embedding=tf.nn.embedding_lookup(word_embedding,self.tf_x)
        input_embedding=tf.nn.dropout(input_embedding,keep_prob=self.keep_prob)

        lstm_fw_cell=tf.nn.rnn_cell.LSTMCell(self.EMBEDDING_DIM,forget_bias=1.0,state_is_tuple=True)
        lstm_bw_cell=tf.nn.rnn_cell.LSTMCell(self.EMBEDDING_DIM,forget_bias=1.0,state_is_tuple=True)
        (output_fw,output_bw),states=tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell,
            lstm_bw_cell,
            input_embedding,
            dtype=tf.float32,
            time_major=False,
            scope=None
        )

        bilstm_out=tf.concat([output_fw,output_bw],axis=2)
        bilstm_out=tf.reshape(bilstm_out,shape=[-1,self.EMBEDDING_DIM*2])
        W=tf.get_variable('W',shape=[self.EMBEDDING_DIM*2,self.TAG_SIZE],dtype=tf.float32)
        b=tf.get_variable('b',shape=[self.TAG_SIZE,],dtype=tf.float32)
        BiLSTM_out=tf.nn.tanh(tf.matmul(bilstm_out,W)+b)
        BiLSTM_out=tf.reshape(BiLSTM_out,shape=[-1,self.SEN_LEN,self.TAG_SIZE])

        #CRF
        log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(BiLSTM_out,self.tf_y, tf.tile(np.array([self.SEN_LEN]),np.array([self.BATCH_SIZE])))
        self.loss=tf.reduce_mean(-log_likelihood)
        self.viterbi_sequence,viterbi_score=tf.contrib.crf.crf_decode(BiLSTM_out,transition_params,tf.tile(np.array([self.SEN_LEN]),np.array([self.BATCH_SIZE])))

        self.train_op=tf.train.AdamOptimizer(self.LR).minimize(self.loss)
