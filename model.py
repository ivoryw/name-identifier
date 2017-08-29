import numpy as np
import scipy as sp
from math import ceil

class log_reg:
    def __init__(self, size, batch_size=1000, alpha=0.2):
        self.batch_size = batch_size
        self.alpha = alpha 
        self.theta = np.full((size,1), -1.0)

    def fit(self, X, y):
        X = X.asformat("csr")
        X_batches = [X[i:i + self.batch_size, :] for i in range(0, X.shape[0] - self.batch_size, self.batch_size)]
        y_batches = [y[i:i + self.batch_size] for i in range(0, len(y) - self.batch_size, self.batch_size)]

        for X_batch, y_batch in zip(X_batches, y_batches):
            y_batch = np.array(y_batch).reshape(len(y_batch),1)
            h = self.__sig(X_batch.dot(self.theta))
            grad = X_batch.transpose().dot(h - y_batch)
            self.theta -= self.alpha * grad

    def predict(self, X):
        P = self.__sig(X.dot(self.theta))
        print P
        if(P < 0.5):
            return 0
        else:
            return 1
            
    def __sig(self, x):
        return 1 / (1 + np.exp(-x))

    def score(self, X, Y):
        s = 0
        for x, y in zip(X, Y):
            P = self.__sig(x.dot(self.theta))
            if P < 0.5:
                r = 0 
            else:
                r = 1 
            if r == y:
                s += 1
        return float(s)/len(Y)

