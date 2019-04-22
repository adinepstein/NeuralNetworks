import Utils
import numpy as np
import pylab
SEED=1234
L2_d=0.0001
L2 = True
DROPOUT=True
DO_PROP = 0.5
LR= 0.001
EPOCHS=100
MINIBATCH_SIZE=20
np.random.seed(SEED)

class nn_layer:

    def __init__(self,input_size,output_size,activation="relu"):
        self.input_size=input_size
        self.output_size=output_size
        self.activation=activation
        self.W = np.random.rand(input_size,output_size)
        self.b = np.random.rand(output_size)
        self.activated_output=0
        self.input=0

    def activate_input(self,input):
        self.input=input
        self.activated_output= self.predict(input)
        return self.activated_output

    def predict(self,input):
        z = np.dot(input, self.W) + self.b
        if self.activation == "relu":
            z = Utils.relu(z)
        elif self.activation == "sigmoid":
            z = Utils.sigmoid(z)
        elif self.activation == "tanh":
            z = Utils.tanh(z)
        elif self.activation == "softmax":
            z = Utils.softmax(z)
        return z

    def activate_derivative(self):
        if self.activation=="relu":
            return Utils.relu_derivative(self.activated_output)
        elif self.activation =="sigmoid":
            return Utils.sigmoid_derivative(self.activated_output)
        elif self.activation=="tanh":
            return Utils.tanh_derivative(self.activated_output)

    def get_mask(self,prop,num_examples):
        self.mask= np.random.binomial(1,1-prop,(num_examples,self.output_size))
        return self.mask


class model:
    def __init__(self):
        self.layers=[]


    def add_layer(self,layer):
        self.layers.append(layer)

    def forward(self,input,dropout_prop,dropout):
        for l in self.layers:
            input = l.activate_input(input)
            if dropout and l!=self.layers[-1]:
                mask=l.get_mask(dropout_prop,input.shape[0])
                input*=mask
        return input


    def backpropogation(self,y,dropout):
        for layer in reversed(self.layers):
            if layer==self.layers[-1]:
                dcrossent_dzo = layer.activated_output-y
                layer.der_w= np.dot(layer.input.T,dcrossent_dzo)
                layer.der_b = dcrossent_dzo
                next_derivative=np.dot(dcrossent_dzo,layer.W.T)

            else:
                next_derivative = next_derivative * layer.activate_derivative()
                if dropout:
                    next_derivative*=layer.mask
                layer.der_w=np.dot(layer.input.T,next_derivative)
                layer.der_b= next_derivative
                next_derivative= np.dot(next_derivative,layer.W.T)

    def update_weights(self,lr,l2_d=1, l_2=False):
        for layer in self.layers:
            layer.der_b=np.mean(layer.der_b,axis=0)
            if l_2:
                layer.W = layer.W - lr * (layer.der_w +l2_d*layer.W)
                layer.b = layer.b - lr * (layer.der_b + l2_d*layer.b)
            else:
                layer.W=layer.W- lr*layer.der_w
                layer.b = layer.b - lr*layer.der_b


    def train(self,train_X,train_y,dev_X,dev_y,batch_size,epochs,dropout_prop=0.5,dropout=True):
        train_loss_list= []
        dev_loss_list = []
        train_accuracy_list = []
        dev_accuracy_list = []
        for epoch in range(epochs):
            train_loss = 0.0
            train_accuracy = 0.0
            dev_loss =0.0
            dev_accuracy=0.0
            numOfRums=0
            for i in range(0,train_X.shape[0],batch_size):
                train_x_mini= train_X[i:i+batch_size]
                train_y_mini = train_y[i:i+batch_size]
                train_output = self.forward(train_x_mini,dropout_prop,dropout)
                train_loss+=Utils.cross_enropy(train_y_mini,train_output)
                train_accuracy+= self.calculate_accuracy(train_output,train_y_mini)
                dev_output = self.predict(dev_X, dropout_prop, dropout)
                dev_loss += Utils.cross_enropy(dev_y,dev_output)
                dev_accuracy +=self.calculate_accuracy(dev_output, dev_y)
                self.backpropogation(train_y_mini, dropout)
                self.update_weights(LR, L2_d, L2)
                numOfRums+=1
            train_loss_list.append(train_loss/numOfRums)
            train_accuracy_list.append(train_accuracy/numOfRums)
            dev_loss_list.append(dev_loss/numOfRums)
            dev_accuracy_list.append(dev_accuracy/numOfRums)
            self.print_loss_accuracy(epoch,train_loss_list[epoch],dev_loss_list[epoch],train_accuracy_list[epoch],dev_accuracy_list[epoch])



    def print_loss_accuracy(self,epoc,train_loss,dev_loss,train_acc,dev_acc):
        print("epoc #" + str(epoc+1) + ": train_loss: %.3f , dev_loss: %.3f, train_accuracy: %.3f, dev_accuracy: %.3f" % (train_loss,dev_loss,train_acc,dev_acc))

    def predict(self,x,dropout_prop=0.5,dropout=True):
        for l in self.layers:
            if dropout and l!=self.layers[-1]:
                l.W = l.W*(1-dropout_prop)
                l.b = l.b*(1-dropout_prop)
        for l in self.layers:
            x = l.predict(x)
        return x

    def calculate_accuracy(self,y_hat,y):
        y= np.argmax(y,axis=1)
        y_hat = np.argmax(y_hat,axis=1)
        currect=0
        for i in range(y.__len__()):
            if y[i]==y_hat[i]:
                currect+=1
        accuracy= currect/y.__len__()
        return accuracy


    def plotdata(self,title,x_axis_name, y_axis_name, label1,data1,label2=None,data2=None):
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

    def create_examples_and_labels(self,num_examples,num_features):
        a_examples = np.random.randn(num_examples, num_features) + np.array([5, 5])
        b_examples = np.random.randn(num_examples, num_features) + np.array([5, -5])
        c_examples = np.random.randn(num_examples, num_features) + np.array([-5, -5])
        d_examples = np.random.rand(num_examples, num_features) + np.array([-5, 5])
        examples = np.vstack([a_examples, b_examples, c_examples, d_examples])
        labels = np.array([0]*num_examples + [1]*num_examples   + [2]* num_examples + [3]*num_examples)
        y_one_hot = np.zeros([4 * num_examples, 4])
        for i in range(4 * num_examples):
            y_one_hot[i, labels[i]] = 1
        return examples,y_one_hot

if __name__ == '__main__':
    model=model()
    model.add_layer(nn_layer(2,15,"relu"))
    model.add_layer(nn_layer(15,4,"softmax"))
    train_x,train_y=model.create_examples_and_labels(1000,2)
    dev_x,dev_y= model.create_examples_and_labels(100,2)
    model.train(train_x,train_y,dev_x,dev_y,MINIBATCH_SIZE,EPOCHS,DO_PROP,DROPOUT)



