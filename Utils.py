import numpy as np

def softmax(x):
    ex = np.exp(x-np.max(x,axis=1,keepdims=True))
    return ex/ex.sum(axis=1,keepdims=True)

def cross_enropy(y_one_hot,y_hat):
    loss = -np.sum(y_one_hot*np.log(y_hat))/y_hat.shape[0]
    return loss

def cross_entropy_softmax_derivative(y_one_hot,y_hat):
    return y_hat-y_one_hot

def relu(x):
    return x * (x>0)

def relu_derivative(x):
    return 1. * (x>0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.- x**2

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1. - x)

def cross_entropy(y_one_hot,y_hat):
    loss= np.mean(np.sum(y_one_hot*y_hat,axis=1))
    loss *= -1
    return loss