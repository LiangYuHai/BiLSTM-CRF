def get_ner_word_and_tag(x,y,id2word,id2tag,res=[]):
    ner=[]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]==0 or y[i][j]==0:
                continue
            elif id2tag[y[i][j]][0]=='B':
                ner=[id2word[x[i][j]]+'/'+id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0]=='M' and len(ner)!=0 and ner[-1].split('/')[-1][1:]==id2tag[y[i][j]][1:]:
                ner.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=='E' and len(ner)!=0 and ner[-1].split('/')[-1][1:]==id2tag[y[i][j]][1:]:
                ner.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                res.append(ner)
                ner=[]
            else:
                ner=[]
    return res

def get_word_and_tag(x,y,id2word,id2tag):
    res=[]
    ner=''
    for i in range(len(x)):
        for j in range(len(x[0])):
            if y[i][j]==0 or x[i][j]==0:
                continue
            elif id2tag[y[i][j]][0]=='B':
                ner=id2tag[y[i][j]][2:]+id2word[x[i][j]]
            elif id2tag[y[i][j]][0]=='M':
                ner+=id2word[x[i][j]]
            elif id2tag[y[i][j]][0]=='E':
                ner+=id2word[x[i][j]]
                res.append(ner)
                ner=''
            else:
                ner=''
    return res

def padding(sentence_id):
    if len(sentence_id)>=60:
        return sentence_id[:60]
    else:
        return sentence_id.extend([0]*(60-len(sentence_id)))