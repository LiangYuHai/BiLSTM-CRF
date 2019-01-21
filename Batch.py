import numpy as np
class BatchGenerator():
    def __init__(self,x,y,shuffle=False):
        if not isinstance(x,np.ndarray):
            x=np.asarray(x)
        if not isinstance(y,np.ndarray):
            y=np.asarray(y)
        self.X=x
        self.Y=y
        self.shuffle=shuffle
        self.num_sentence=self.X.shape[0]
        self.epoch_completed=0
        self.index_sentence=0
        if shuffle:
            new_index=np.random.permutation(self.num_sentence)
            self.X=self.X[new_index]
            self.Y=self.y[new_index]
    @property
    def x(self):
        return self.X
    @property
    def y(self):
        return self.Y

    @property
    def epoch_completed_num(self):
        return self.epoch_completed

    @property
    def Num_Sentence(self):
        return self.num_sentence

    def next_batch(self,batch_size):
        start=self.index_sentence
        self.index_sentence+=batch_size
        if self.index_sentence>self.num_sentence:
            self.epoch_completed+=1
            if self.shuffle:
                new_index=np.random.permutation(self.num_sentence)
                self.X=self.X[new_index]
                self.Y=self.Y[new_index]
            start=0
            self.index_sentence=batch_size
        end=self.index_sentence
        return self.X[start:end],self.Y[start:end]