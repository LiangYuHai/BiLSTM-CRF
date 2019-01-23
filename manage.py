import sys
import tensorflow as tf
from Batch import BatchGenerator
import pickle
import numpy as np
from BiLSTM_CRF import Model
import re
from utils import get_word_and_tag,get_ner_word_and_tag
with open('../bosondata.pkl','rb') as inp:
    word2id=pickle.load(inp)
    id2word=pickle.load(inp)
    tag2id=pickle.load(inp)
    id2tag=pickle.load(inp)
    x_train=pickle.load(inp)
    x_test=pickle.load(inp)
    y_train=pickle.load(inp)
    y_test=pickle.load(inp)
print(len(word2id))
print(word2id)
print(tag2id)
config={}
config['LR']=0.01
config['EMBEDDING_DIM']=100
config['SEN_LEN']=len(x_train[0])
config['EMBEDDING_SIZE']=len(word2id)+1
config['TAG_SIZE']=len(tag2id)+1
config['BATCH_SIZE'] = 32
if len(sys.argv)==2 and sys.argv[1]=='train':
    model=Model()
    model.init(config)
    model.build_net()
    EPOCH = 20
    BATCH_SIZE = config['BATCH_SIZE']
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
        for batch in range(num_train_batch_per_epoch):
            batch_x,batch_y=train_data.next_batch(BATCH_SIZE)
            loss_,pred,_=sess.run([model.loss,model.viterbi_sequence,model.train_op],feed_dict={model.tf_x:batch_x,model.tf_y:batch_y,model.keep_prob:0.5})
            if batch %200==0:
                accuracy=np.equal(pred, batch_y)
                accuracy=accuracy.astype(np.int32)
                accuracy_rate = np.mean(accuracy)
                print('epoch:',epoch,'| batch:',batch,'| loss:%.4f'%loss_,'| accuracy_rate:%.4f'%accuracy_rate)

        if epoch%3==0:
            pred_ner_word_and_tag=[]
            actu_ner_word_and_tag=[]
            for batch in range(num_train_batch_per_epoch):
                batch_x,batch_y=train_data.next_batch(BATCH_SIZE)
                pred=sess.run(model.viterbi_sequence,feed_dict={model.tf_x:batch_x,model.tf_y:batch_y,model.keep_prob:1})
                pred_ner_word_and_tag = get_ner_word_and_tag(batch_x, pred, id2word, id2tag, pred_ner_word_and_tag)
                actu_ner_word_and_tag = get_ner_word_and_tag(batch_x, batch_y, id2word, id2tag, actu_ner_word_and_tag)
            print(len(pred_ner_word_and_tag))
            print(len(actu_ner_word_and_tag))
            correct=[i for i in pred_ner_word_and_tag if i in actu_ner_word_and_tag]
            if len(correct)!=0:
                ner_accuracy_rate=float(len(correct))/len(pred_ner_word_and_tag)
                ner_recall_rate=float(len(correct))/len(actu_ner_word_and_tag)
                F1=2*ner_accuracy_rate*ner_recall_rate/(ner_accuracy_rate+ner_recall_rate)
                print('train:')
                print('ner_accuracy_rate:%.4f'%ner_accuracy_rate)
                print('ner_recall_rate:%.4f'%ner_recall_rate)
                print('F1:%.4f'%F1)
            else:
                print('test:')
                print('ner_accuracy_rate:0')

            for batch in range(num_test_batch_per_epoch):
                batch_x,batch_y=test_data.next_batch(BATCH_SIZE)
                pred=sess.run(model.viterbi_sequence,feed_dict={model.tf_x:batch_x,model.tf_y:batch_y,model.keep_prob:1})
                pred_ner_word_and_tag.append(get_ner_word_and_tag(batch_x,pred,id2word,id2tag,pred_ner_word_and_tag))
                actu_ner_word_and_tag.append(get_ner_word_and_tag(batch_x,batch_y,id2word,id2tag,pred_ner_word_and_tag))
            correct=[i for i in pred_ner_word_and_tag if i in actu_ner_word_and_tag]
            if len(correct)!=0:
                ner_accuracy_rate=float(len(correct))/len(pred_ner_word_and_tag)
                ner_recall_rate=float(len(correct))/len(actu_ner_word_and_tag)
                F1=2*ner_accuracy_rate*ner_recall_rate/(ner_accuracy_rate+ner_recall_rate)
                print('test:')
                print('ner_accuracy_rate:%.4f'%ner_accuracy_rate)
                print('ner_recall_rate:%.4f'%ner_recall_rate)
                print('F1:%.4f'%F1)
                if F1>max_F1:
                    max_F1=F1
                    saver.save(sess,'../Ner/model/'+str(epoch)+'.ckpt')
            else:
                print('test:')
                print('ner_accuracy_rate:0')

from utils import padding
if len(sys.argv)==2 and sys.argv[1]=='test':
    model=Model()
    config['BATCH_SIZE'] = 1 #想输入多少句话就设为多少，以标点符号为断句。
    model.init(config)
    model.build_net()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state('../Ner/model')
    if not ckpt:
        print('model not found,please train model firstly')
    else:
        path = ckpt.model_checkpoint_path
        saver.restore(sess, save_path=path)
    while True:
        original_test=input('input:')
        test=re.split(u'[，。！？、‘’“”（）]',original_test)
        test_id = []
        for sentence in test:
            if sentence!=' ' and sentence!='  ':
                sentence_id=[]
                for word in sentence:
                    if word in word2id.index:
                        sentence_id.append(word2id[word])
                    else:
                        sentence_id.append(word2id["unknow"])
                if len(sentence_id)>=60:
                    sentence_id=sentence_id[:60]
                else:
                    sentence_id.extend([0]*(60-len(sentence_id)))
                test_id.append(sentence_id)

        pred=sess.run(model.viterbi_sequence,feed_dict={model.tf_x:test_id,model.keep_prob:1})
        res=get_word_and_tag(test_id,pred,id2word,id2tag)
        print(original_test)
        if len(res)!=0:
            for i in res:
                print(i)
        else:
            print('未识别出任何实体。')
