import numpy as np

class Hebbian():
    def __init__(self,dim_from,dim_to,lr,time_limit,rule='sanger',change_lr='never'):
        np.random.seed(42)
        self.W = np.array(np.random.rand(dim_to, dim_from)*2-1,dtype=np.double)
        self.lr, self.time_limit,self.rule,self.change_lr,self.first_lr = lr, time_limit,rule,change_lr, lr

    def fit_one(self, X):
        if self.rule == 'sanger':
            Y = np.dot(self.W, X)
            dW = (np.outer(Y,X) - np.dot(np.tril(np.outer(Y,Y)),self.W)) * self.lr
            self.W += dW
        if self.rule == 'oja':
            W = self.W
            Y = np.dot(self.W, X)
            dW = np.outer(Y, X-np.dot( Y, W)) * self.lr
            self.W += dW

    def fit(self,X):
        last_ortogonalities = np.ones(100) * 100
        for t in range(1,self.time_limit):
            if self.change_lr == 'always':
                self.lr = self.first_lr / t
            for i in range(X.shape[0]):
                self.fit_one(X[i])
            
            o = np.sum(np.abs( np.dot( self.W, self.W.T) - np.identity(len(self.W)) ))/2
            last_ortogonalities[ t % len(last_ortogonalities)] = o
            if self.change_lr == 'when_stuck':
                if np.all(last_ortogonalities[:10] - o < 1e-6):
                    self.lr = self.lr / 10
            if o < 0.01:
                print('convergio')
                break
        return self
    
    def transform(self,X):
        return np.dot(X, self.W.T)