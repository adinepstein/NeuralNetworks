import numpy as np
import pylab
import pickle
import datetime


def softmax(x):
    ex = np.exp(x-np.max(x,axis=1,keepdims=True))
    sum_ex = ex/ex.sum(axis=1,keepdims=True)
    return sum_ex

def cross_enropy(y_one_hot,y_hat):
    size = y_hat.shape[0]
    sum = np.sum(y_hat)/size
    print(sum)
    loss = -(1.0/size) * np.sum(y_one_hot*np.log(y_hat)+(1-y_one_hot)*np.log(1-y_hat))
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

def readData(filePath,test=False):
    data = np.genfromtxt(filePath, delimiter=',')
    labels = data[:,0]
    features = data[:,1:]
    if test:
        return features
    else:
        y_one_hot=np.zeros([labels.size(),10])
        for i in range(labels.size()):
            y_one_hot[i,labels[i]-1]=1
        return features,y_one_hot

def plotdata(title,x_axis_name, y_axis_name, label1,data1,label2=None,data2=None):
    size = len(data1)
    x = []
    for i in range(size):
        x.append((i))
    pylab.plot(x, data1, "-r", label=label1)
    if data2!=None:
        pylab.plot(x,data2,"-b",label=label2)
    pylab.legend()
    pylab.title(title)
    pylab.xlabel(x_axis_name)
    pylab.ylabel(y_axis_name)
    pylab.show()

def save_to_pickle(path,csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    with open(path, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def load_data_pickle(path,test=False):
    with open(path, 'rb') as infile:
        data = pickle.load(infile)
    labels = data[:, 0]
    features = data[:, 1:]
    if test:
        return features
    else:
        y_one_hot = np.zeros([labels.__len__(), 10])
        for i in range(labels.__len__()):
            y_one_hot[i, int(labels[i]) - 1] = 1
        return features, y_one_hot

if __name__ == '__main__':
    print(datetime.datetime.now())
    save_to_pickle("test.pkl","test.csv")
    print(datetime.datetime.now())
    save_to_pickle("validate.pkl", "validate.csv")
    print(datetime.datetime.now())
    save_to_pickle("train.pkl", "train.csv")
    print(datetime.datetime.now())

