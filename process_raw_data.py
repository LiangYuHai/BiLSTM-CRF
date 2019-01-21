import codecs
import pandas as pd
import numpy as np
import re

def raw2tag():
    input=codecs.open('./raw_data.txt','r','utf-8')
    output=codecs.open('./word_tag.txt','w','utf-8')
    for line in input.readlines():
        line=line.strip()
        i=0
        while i<len(line):
            if line[i]=='{':
                i+=2
                temp=''
                while line[i]!='}':
                    temp+=line[i]
                    i+=1
                i+=2
                word_tag=temp.split(':')
                output.write(word_tag[1][0]+'/B_'+word_tag[0]+' ')
                for j in word_tag[1][1:-1]:
                    output.write(j+'/M_'+word_tag[0]+' ')
                output.write(word_tag[1][-1]+'/E_'+word_tag[0]+' ')
            else:
                output.write(line[i]+'/O'+' ')
                i+=1
        output.write('\n')
    input.close()
    output.close()

def get_rid_of_punctuation():
    input=codecs.open('./word_tag.txt','r','utf-8')
    output=codecs.open('./sentences.txt','w','utf-8')
    for line in input.readlines():
        line=line.strip()
        sentences = re.split('[，。！？、‘’“”（）]/[O]',line)
        for sentence in sentences:
            if sentence !=" ":
                output.write(sentence.strip()+'\n')
    input.close()
    output.close()

def data2pkl():
    words2d=[]
    label2d=[]
    words=[]
    tags=set()
    input=codecs.open('./sentences.txt','r','utf-8')
    for line in input.readlines():
        line=line.strip()
        words_tags=line.split()
        words_per_line=[]
        tags_per_line=[]
        numNotO=0
        for word_tag in words_tags:
            word_tag=word_tag.split('/')
            words.append(word_tag[0])
            words_per_line.append(word_tag[0])
            tags.add(word_tag[1])
            tags_per_line.append(word_tag[1])
            if word_tag[1]!='O':
                numNotO+=1
        if numNotO!=0:
            words2d.append(words_per_line)
            label2d.append(tags_per_line)

    input.close()
    from collections import Counter
    words_counts=Counter(words)
    wordset=list(words_counts.keys())
    wordidset=range(1,len(wordset)+1)
    word2id=pd.Series(wordidset,index=wordset)
    id2word=pd.Series(wordset,index=wordidset)
    word2id['unknow']=len(word2id)+1

    tagset=list(tags)
    tagidset=range(1,len(tagset)+1)
    tag2id=pd.Series(tagidset,index=tagset)
    id2tag=pd.Series(tagset,index=tagidset)

    df_data=pd.DataFrame({'word':words2d,'tag':label2d})

    max_len=60
    def word_id_padding(words):
        ids=list(word2id[words])
        if len(ids)>=max_len:
            return ids[:max_len]
        else:
            ids.extend([0]*(max_len-len(ids)))
            return ids
    def tag_id_padding(tags):
        ids=list(tag2id[tags])
        if len(ids)>=max_len:
            return ids[:max_len]
        else:
            ids.extend([0]*(max_len-len(ids)))
            return ids
    df_data['word_id']=df_data['word'].apply(word_id_padding)
    df_data['tag_id']=df_data['tag'].apply(tag_id_padding)
    word_id=np.asarray(list(df_data['word_id'].values))
    tag_id=np.asarray(list(df_data['tag_id'].values))

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(word_id, tag_id, test_size=0.2,random_state=43)


    import pickle
    import os
    with open('../bosondata.pkl','wb') as outpkl:
        pickle.dump(word2id,outpkl)
        pickle.dump(id2word,outpkl)
        pickle.dump(tag2id,outpkl)
        pickle.dump(id2tag,outpkl)
        pickle.dump(x_train,outpkl)
        pickle.dump(x_test,outpkl)
        pickle.dump(y_train,outpkl)
        pickle.dump(y_test,outpkl)
    print('Finish pickling data')

raw2tag()
get_rid_of_punctuation()
data2pkl()



