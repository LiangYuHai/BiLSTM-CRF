import tensorflow as tf
import numpy as np
import matplotlib as plt
from Batch import BatchGenerator
import pickle
with open('../bosondata.pkl','rb') as inp:
    word2id=pickle.load(inp)
    id2word=pickle.load(inp)
    tag2id=pickle.load(inp)
    id2tag=pickle.load(inp)
    x_train=pickle.load(inp)
    x_test=pickle.load(inp)
    y_train=pickle.load(inp)
    y_test=pickle.load(inp)

LR=0.01
BATCH_SIZE=32
EMBEDDING_DIM=100
SEN_LEN=len(x_train[0])
EMBEDDING_SIZE=len(word2id)+1 #还要+1? 空白to 0
TAG_SIZE=len(tag2id)+1
EPOCH=3
# config["pretrained"]=False


tf_x=tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE,SEN_LEN])
tf_y=tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE,SEN_LEN])
keep_prob=tf.placeholder(tf.float32)
word_embedding=tf.get_variable("word_embedding",[EMBEDDING_SIZE,EMBEDDING_DIM],dtype=tf.float32)
input_embedding=tf.nn.embedding_lookup(word_embedding,tf_x)
input_embedding=tf.nn.dropout(input_embedding,keep_prob=keep_prob)

lstm_fw_cell=tf.nn.rnn_cell.LSTMCell(EMBEDDING_DIM,forget_bias=1.0,state_is_tuple=True)
lstm_bw_cell=tf.nn.rnn_cell.LSTMCell(EMBEDDING_DIM,forget_bias=1.0,state_is_tuple=True)
(output_fw,output_bw),states=tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell,
    lstm_bw_cell,
    input_embedding,
    dtype=tf.float32,
    time_major=False,
    scope=None
)

bilstm_out=tf.concat([output_fw,output_bw],axis=2)
W=tf.get_variable('W',shape=[BATCH_SIZE,EMBEDDING_DIM*2,TAG_SIZE],dtype=tf.float32)
b=tf.get_variable('b',shape=[BATCH_SIZE,SEN_LEN,TAG_SIZE],dtype=tf.float32,initializer=tf.zeros_initializer())
BiLSTM_out=tf.nn.tanh(tf.matmul(bilstm_out,W)+b)

#CRF
log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(BiLSTM_out,tf_y, tf.tile(np.array([SEN_LEN]),np.array([BATCH_SIZE])))
loss=tf.reduce_mean(-log_likelihood)

viterbi_sequence,viterbi_score=tf.contrib.crf.crf_decode(BiLSTM_out,transition_params,tf.tile(np.array([SEN_LEN]),np.array([BATCH_SIZE])))

train_op=tf.train.AdamOptimizer(LR).minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver(max_to_keep=1)

train_data=BatchGenerator(x_train,y_train,shuffle=True)
test_data=BatchGenerator(x_test,y_test,shuffle=False)

for epoch in range(EPOCH):
    num_train_batch_per_epoch=int(train_data.x.shape[0]/BATCH_SIZE)
    num_test_batch_per_epoch=int(test_data.x.shape[0]/BATCH_SIZE)
    max_F1=0
    for batch in range(3):
        batch_x,batch_y=train_data.next_batch(BATCH_SIZE)
        loss_,pred,_=sess.run([loss,viterbi_sequence,train_op],feed_dict={tf_x:batch_x,tf_y:batch_y,keep_prob:0.5})
        if batch %200==0:
            accuracy=0
            for i in range(len(pred)):
                # accuracy=np.cast(np.equal(pred,batch_y))
                # accuracy_rate=np.mean(accuracy)
                for j in range(len(pred[0])):
                    if pred[i][j]==batch_y[i][j]:
                        accuracy+=1
            accuracy_rate=float(accuracy)/(len(pred)*len(pred[0]))
            print('epoch:',epoch,'| batch:',batch,'| loss:%.4f'%loss_,'| accuracy_rate:%.4f'%accuracy_rate)

    if epoch%3==0:
        from utils import get_ner_word_and_tag
        pred_ner_word_and_tag=[]
        actu_ner_word_and_tag=[]
        for batch in range(num_train_batch_per_epoch):
            batch_x,batch_y=train_data.next_batch(BATCH_SIZE)
            pred=sess.run(viterbi_sequence,feed_dict={tf_x:batch_x,tf_y:batch_y,keep_prob:1})
            # pre = pre[0]
            pred_ner_word_and_tag = get_ner_word_and_tag(batch_x, pred, id2word, id2tag, pred_ner_word_and_tag)
            actu_ner_word_and_tag = get_ner_word_and_tag(batch_x, batch_y, id2word, id2tag, actu_ner_word_and_tag)
            # pred_ner_word_and_tag.extend(get_ner_word_and_tag(batch_x,pred,id2word,id2tag))
            # actu_ner_word_and_tag.extend(get_ner_word_and_tag(batch_x,batch_y,id2word,id2tag))
            print(pred[0].shape)
            print(batch)
            #id2tag[pred]
            #id2tag[batch_y]
        # correct=[i for i in pred_ner_word_and_tag if i in actu_ner_word_and_tag]
        # if len(correct)!=0:
        #     ner_accuracy_rate=float(len(correct))/len(pred_ner_word_and_tag)
        #     ner_recall_rate=float(len(correct))/len(actu_ner_word_and_tag)
        #     F1=2*ner_accuracy_rate*ner_recall_rate/(ner_accuracy_rate+ner_recall_rate)
        #     print('train:')
        #     print('ner_accuracy_rate:%.4f'%ner_accuracy_rate)
        #     print('ner_recall_rate:%.4f'%ner_recall_rate)
        #     print('F1:%.4f'%F1)
        # else:
        #     print('test:')
        #     print('ner_accuracy_rate:0')

        # for batch in range(num_test_batch_per_epoch):
        #     batch_x,batch_y=test_data.next_batch(BATCH_SIZE)
        #     pred=sess.run(viterbi_sequence,feed_dict={tf_x:batch_x,tf_y:batch_y,keep_prob:1})
        #     pred_ner_word_and_tag.extend(get_ner_word_and_tag(batch_x,pred,id2word,id2tag))
        #     actu_ner_word_and_tag.extend(get_ner_word_and_tag(batch_x,batch_y,id2word,id2tag))
        #     #id2tag[pred]
        #     print(pred.shape)
        #     print(batch_x.shape)
        #     print(batch_y.shape)
        #     # for i in range(batch_y.shape[0]):
        #     #     actu_ner_word_and_tag.append(id2tag[batch_y[i]])
        # import operator
        # correct=0
        # for i in pred_ner_word_and_tag:
        #     for j in actu_ner_word_and_tag:
        #         if operator.eq(i,j):
        #             correct+=1
        #             break
        # # correct=[i for i in pred_ner_word_and_tag if i in actu_ner_word_and_tag]
        # print(correct)
        # if len(correct)!=0:
        #     ner_accuracy_rate=float(correct)/len(pred_ner_word_and_tag)
        #     ner_recall_rate=float(correct)/len(actu_ner_word_and_tag)
        #     F1=2*ner_accuracy_rate*ner_recall_rate/(ner_accuracy_rate+ner_recall_rate)
        #     print('test:')
        #     print('ner_accuracy_rate:%.4f'%ner_accuracy_rate)
        #     print('ner_recall_rate:%.4f'%ner_recall_rate)
        #     print('F1:%.4f'%F1)
        #     if F1>max_F1:
        #         saver.save(sess,'../model/'+str(epoch)+'.ckpt')
        # else:
        #     print('test:')
        #     print('ner_accuracy_rate:0')
