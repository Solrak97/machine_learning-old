import pandas as pd
import numpy as np
import sys

def MSE(y_true, y_predict) -> float:
    n = y_true.shape[0]
    if n != y_predict.shape[0]:
        raise Exception("true and predict sizes don't match")
    
    else:
        return np.sum(((y_true - y_predict) ** 2)/n)
   

def MAE(y_true, y_predict) -> float:
    n = y_true.shape[0]
    if n != y_predict.shape[0]:
        raise Exception("true and predict sizes don't match")
    
    else:
        return np.sum(abs(y_predict- y_true)/n)


def score(y_true, y_predict) -> float:
    n = y_true.size
    if n != y_predict.size:
        raise Exception("true and predict sizes don't match")
    
    else:
        return (np.var(y_predict) - np.var(y_true))


class LinearRegression:
    def __init__(self):
        self.c = None
        pass


    def L1_dv(self, _lambda):
        return (_lambda * abs(self.C)) 
    

    def L2_dv(self, _lambda):
        return (_lambda * self.C ** 2)

    
    def elasticnet_dv(self, _lambda):
        return ((_lambda * self.C ** 2) + ((1 - _lambda) * abs(self.C))) 


    def fit(self, x, y, max_epochs=100, threshold=0.01, learning_rate=0.001,
        momentum=0, decay=0, error='mse', regularization='none', _lambda=0):
        
        # Revisar que X, Y son del mismo tamaño
        n = len(x.index)
        if n != len(y.index):
            raise Exception("x and y sizes don't match")

        # Inserción del bias en la matriz de datos
        x.insert(loc=0, column = 'Bias', value=[1 for i in range(n)])

        # Variables necesarias para calcular la regresión
        self.C = np.random.normal(size=len(x.columns))
        self.C.shape = (self.C.shape[0], 1)
        err_func = MAE if error == 'mae' else MSE
        
        if regularization == 'l2' or regularization == 'ridge':
            regulator = self.L1_dv
        elif regularization == 'elastic-net':
            regulator = self.elasticnet_dv
        else:
            regulator = self.L1_dv

        X = x.to_numpy()
        Y = y.to_numpy()
        Y.shape = (Y.shape[0], 1)

        # Variables para el control de las epocas
        epoch = 0
        it_threshold = sys.maxsize 

        error = 0    
        last_dv = 0
        eta = learning_rate
        error_list = []

        # Epocas
        while(epoch < max_epochs and it_threshold > threshold):
            e = (np.matmul(X, self.C) - Y)
            last_dv = (np.matmul(e.transpose(), X).transpose() / X.shape[0])
            self.C -=  eta * (last_dv + (momentum * last_dv)) + regulator(_lambda)
            eta /= (1 + decay)

            n_error = err_func(np.matmul(X, self.C), Y)
            if error != 0:
                it_threshold = abs(1 - (n_error / error))

            error = n_error
            error_list.append(error)

            epoch += 1
        
        return error_list


    def predict(self, x):
        n = x.shape[0]
        x.insert(loc=0, column = 'Bias', value=[1 for i in range(n)])
        return np.matmul(x.to_numpy(), self.C)[:,0]