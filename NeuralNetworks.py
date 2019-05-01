import Utils
import numpy as np
import pylab
import pickle
SEED=1234
L2_d=0.001
L2 = True
DROPOUT=True
DO_PROP = 0.3
LR= 0.01
EPOCHS=100
MINIBATCH_SIZE=1
np.random.seed(SEED)
NUM_ITERATIONS_DROP_LR=10
DROP_LR=0.5
class nn_layer:

    def __init__(self,input_size,output_size,activation="relu"):
        self.input_size=input_size
        self.output_size=output_size
        self.activation=activation
        self.W = np.random.uniform(low=-1/input_size,high=1/input_size,size=(input_size,output_size))
        self.b = np.zeros(output_size)
        self.activated_output=0
        self.input=0

    def activate_input(self,input):
        self.input=input
        self.z= np.dot(input, self.W) + self.b
        self.activated_output= self.activate(self.z)
        return self.activated_output

    def predict(self,x,dropout_prop=0.5,dropout=True):
        if dropout:
            if self.activation=="softmax":
                x = np.dot(x, self.W) + self.b
            else:
                x = np.dot(x, self.W*(1-dropout_prop)) + self.b*(1-dropout_prop)
        else:
            x = np.dot(x, self.W) + self.b
        return self.activate(x)

    def activate(self,z):
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
            return Utils.relu_derivative(self.z)
        elif self.activation =="sigmoid":
            return Utils.sigmoid_derivative(self.z)
        elif self.activation=="tanh":
            return Utils.tanh_derivative(self.z)

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
                layer.b = layer.b - lr * layer.der_b
            else:
                layer.W=layer.W- lr*layer.der_w
                layer.b = layer.b - lr*layer.der_b


    def train(self,train_X,train_y,val_X,val_y,batch_size,epochs,lr,dropout_prop=0.5,dropout=True):
        train_loss_list= []
        val_loss_list = []
        train_accuracy_list = []
        val_accuracy_list = []
        for epoch in range(epochs):
            lr = self.decay_lr(lr, epoch)
            train_predict=np.zeros(shape=train_y.shape)
            for i in range(0,train_X.shape[0],batch_size):
                train_x_mini= train_X[i:i+batch_size]
                train_y_mini = train_y[i:i+batch_size]
                t_t=self.forward(train_x_mini,dropout_prop,dropout)
                train_predict[i:i+batch_size] = t_t
                self.backpropogation(train_y_mini, dropout)
                self.update_weights(lr, L2_d, L2)
            val_output = self.predict(val_X, dropout_prop, dropout)
            train_loss_list.append(Utils.cross_enropy(train_y, train_predict))
            train_accuracy_list.append(self.calculate_accuracy(train_predict, train_y))
            val_loss_list.append(Utils.cross_enropy(val_y, val_output))
            val_accuracy_list.append(self.calculate_accuracy(val_output, val_y))
            self.print_loss_accuracy(epoch,train_loss_list[epoch],val_loss_list[epoch],train_accuracy_list[epoch],val_accuracy_list[epoch])
        Utils.plotdata("Loss","# epoch", "loss","train",train_loss_list,"validation",val_loss_list)
        Utils.plotdata("Accuracy", "# epoch", "%", "train", train_accuracy_list, "validation", val_accuracy_list)

    def decay_lr(self,lr,epoc):
        if epoc%NUM_ITERATIONS_DROP_LR==0 and epoc!=0:
            return lr*DROP_LR
        else:
            return lr

    def normalize_train_data(self,data):
        self.avg=np.average(data)
        self.std = np.std(data)
        return (data-self.avg)/self.std

    def print_loss_accuracy(self,epoc,train_loss,dev_loss,train_acc,dev_acc):
        print("epoc #" + str(epoc+1) + ": train_loss: %.3f , dev_loss: %.3f, train_accuracy: %.3f, dev_accuracy: %.3f" % (train_loss,dev_loss,train_acc,dev_acc))

    def predict(self,x,dropout_prop=0.5,dropout=True):
            for l in self.layers:
                x = l.predict(x,dropout_prop,dropout)
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

    def save_model_data(self, pkl_path):
        model_data = []
        model_data.append([self.avg, self.std])
        for layer in self.layers:
            model_data.append([layer.activation, layer.W, layer.b])
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(model_data, outfile, pickle.HIGHEST_PROTOCOL)

    def load_model_data(self, pkl_path):
        with open(pkl_path, 'rb') as infile:
            result = pickle.load(infile)
        num_layers= result.__len__()
        self.avg=result[0][0]
        self.std = result[0][1]
        for i in range(1,num_layers):
            l =result[i]
            activation= l[0]
            W = l[1]
            b =l[2]
            layer =nn_layer(W.shape[0],W.shape[1],activation)
            layer.W=W
            layer.b = b
            self.add_layer(layer)



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
    # model.add_layer(nn_layer(2,6,"relu"))
    # model.add_layer(nn_layer(6,4,"softmax"))
    # train_x,train_y=model.create_examples_and_labels(5,2)
    # val_x,val_y= model.create_examples_and_labels(2,2)
    model.add_layer(nn_layer(3072,32,"relu"))
    model.add_layer(nn_layer(32,10,"softmax"))
    train_x,train_y = Utils.load_data_pickle("train.pkl")
    val_x,val_y = Utils.load_data_pickle("validate.pkl")
    train_x=model.normalize_train_data(train_x)
    val_x=(val_x-model.avg)/model.std
    model.train(train_x,train_y,val_x,val_y,MINIBATCH_SIZE,EPOCHS,LR,DO_PROP,DROPOUT)
    model.save_model_data("test.pkl")



