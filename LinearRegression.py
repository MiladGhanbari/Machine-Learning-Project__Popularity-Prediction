# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:49:39 2020

@author: Milad Ghanbari
"""
from numpy import linalg
import numpy as np


def ClosedForm(x_train,y_train): 
    l = x_train.shape[1]
    W = np.zeros([l,1])
    X1 = linalg.inv(np.matmul(x_train.transpose(),x_train)) 
    X2 = np.matmul(X1, x_train.transpose())
    W = np.matmul(X2, y_train)
    return W


def GradientDescent(x_train, y_train, eps = 0.0001, eta = 0.000001
, beta = 0.000001, printing = True, epochs = 0 ):
    l = x_train.shape[1]
    lr = eta / (1 + beta)
    W_old = np.zeros([l,1])
    W_new = np.zeros([l,1])
    gradient_error = []
    i=0
    X1 = np.matmul(x_train.transpose(),x_train)
    X3 = np.matmul(x_train.transpose(),y_train)
    if(epochs == 0):
        while(True):
            X2 = np.matmul(X1,W_old)
            i+=1
            W_new = W_old - 2 * lr * (X2 - X3)
            lr = lr / (1 + beta)
            error = linalg.norm( (W_new - W_old) , ord = 2)
            gradient_error.append(error)
            W_old = W_new
            if(printing):
                print("Iteration {} and Norm Error: {}".format(i, error))
            if (error < eps ):
                break
            elif (error > 100):
                print("Gradient Diverged!")
                break
    else:
        for i in range(epochs):
            X2 = np.matmul(X1,W_old)
            i+=1
            W_new = W_old - 2 * lr * (X2 - X3)
            error = linalg.norm( (W_new - W_old) , ord = 2)
            gradient_error.append(error)
            W_old = W_new
            if(printing):
                print("Iteration {} and Norm Error: {}".format(i, error))
            if(error > 100):
                break
            
    return W_new, gradient_error

def mse(y_train, y_pred):
    return np.mean((np.square(y_train-y_pred)))