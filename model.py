import numpy as np
import scipy as sp


class LogReg:
    def __init__(self, size, batch_size=1000, alpha=0.2, C=0):
        """
        Initialise the LogReg object
        :param size: Number of samples in the training set to be fitted
        :type size: int
        :param batch_size: Size of batch for mini-batch gradient descent
        :type batch_size: int
        :param alpha: Learning rate for mini-batch gradient descent
        :type alpha: float
        :param C: L2 regularisation constant
        :type C: float
        """
        self.batch_size = batch_size
        self.alpha = alpha
        self.theta = np.full((size, 1), -1.0)
        self.C = C

    def fit(self, X, y):
        """
        Fits a logistic regressor using a labeled data-set using SDG and L2 regularisation
        :param X: Training samples of shape (n_samples, n_features)
        :type X: sp.sparse_matrix
        :param y: A array containing binary labels for each sample, of shape (n_samples)
        :type y: np.ndarray
        :return: None
        """
        X = X.asformat("csr")
        X_batches = [X[i:i + self.batch_size, :] for i in range(0, X.shape[0] - self.batch_size, self.batch_size)]
        y_batches = [y[i:i + self.batch_size] for i in range(0, len(y) - self.batch_size, self.batch_size)]

        for X_batch, y_batch in zip(X_batches, y_batches):
            y_batch = np.array(y_batch).reshape(len(y_batch), 1)
            h = self.__sig(X_batch.dot(self.theta))
            grad = X_batch.transpose().dot(h - y_batch)
            self.theta -= self.alpha * (grad + self.C * self.theta)

    def predict(self, X):
        """
        Predicts labels for X using the fitted logistic regressor
        :param X: A data-set of shape (n_samples, n_features) to be predicted
        :type X: np.ndarray
        :return: Predicted binary labels of shape (n_samples)
        :rtype: list
        """
        P = self.__sig(X.dot(self.theta))
        result = [0 if p < 0.5 else 1 for p in P]
        return result

    def __sig(self, x):
        """
        Vectorised sigmoid function
        :param x: A numpy array to be evaluated
        :type x: np.ndarray
        :return: A numpy array containing the results
        :rtype: np.ndarray
        """
        return 1 / (1 + np.exp(-x))

    def score(self, X, Y):
        """
        Evaluates the mean correct prediction rate for a labelled data-set
        :param X: Samples to be predicted, of shape (n_samples, n_features)
        :type X: sp.sparse_matrix
        :param Y: Binary labels for each sample, of shape (n_samples)
        :type Y: list
        :return: Mean correct prediction rate of X against Y
        :rtype: float
        """
        s = 0.0
        P = self.__sig(X.dot(self.theta))
        for p, y in zip(P, Y):
            if (p >= 0.5) == bool(y):
                s += 1.0
        return s / len(Y)
