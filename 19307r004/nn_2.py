import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

def relu(layer):
    layer[layer < 0] = 0
    return layer

def relu_d(layer):
    layer[layer<=0]=0
    layer[layer>0]=1
    return layer

class Net(object):

    def __init__(self, num_layers, num_units):

        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []
        
        for i in range(num_layers):

            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
#                 print('weights :',i,self.weights[i].shape)
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))
#                 print('weights :',i,self.weights[i].shape)
                

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

    def __call__(self, X):
        a = X
        self.h_states = []
        self.a_states = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i==0:
                self.h_states.append(a) # For input layer, both h and a are same
            else:
                self.h_states.append(h)
            self.a_states.append(a)

            h = np.dot(a, w) + b.T

            if i < len(self.weights)-1:
                a = relu(h)
            else: # No activation for the output layer
                a = h

        self.pred = a
        return self.pred


    def backward(self, X, y, lamda):
        del_W = []
        del_b = []
        dA = 2*(self.pred-y)
        
        for i in reversed(range(self.num_layers+1)):
            m = y.shape[0]
            if i == self.num_layers:
                dZ=dA
            else:
                dZ= dA*relu_d(dA)
            dW = np.dot(dZ.T,self.a_states[i])/m
            dW = dW.T + lamda * (self.weights[i]/m)
            del_W.append(dW)
            
            db = np.sum(dZ.T,axis=1,keepdims = True)/m
            del_b.append(db)
            
            dA_prev = np.dot(self.weights[i],dZ.T)
            dA = dA_prev.T
        del_W.reverse()
        del_b.reverse()
        
        return del_W,del_b


class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


    def step(self, weights, biases, delta_weights, delta_biases):
        new_weights=[]
        new_biases =[]
        for l in range(len(weights)):
            new_weights.append(weights[l] - self.learning_rate*delta_weights[l])
            new_biases.append(biases[l] - self.learning_rate*delta_biases[l])
        return new_weights,new_biases

    def loss_mse(self,y, y_hat):
        return np.mean(np.power(y-y_hat,2))

    def loss_regularization(self,weights, biases):
        sum =0
        for i in range(len(weights)):
            sum += np.sum(np.power(weights[i],2))+np.sum(np.power(biases[i],2))
        return sum

    def loss_fn(self,y, y_hat, weights, biases, lamda):
        loss = self.loss_mse(y,y_hat) + lamda * self.loss_regularization(weights,biases)
        return loss
    
    def rmse(self,y, y_hat):
        return (np.mean(np.power(y-y_hat,2)))**0.5

def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
):


    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        print('epoch :',e)
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            if len(batch_input) != batch_size:
                break
            pred = net(batch_input)

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = optimizer.loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss
#         print(epoch_loss)

            #print(e, i, rmse(batch_target, pred), batch_loss)

        #print(e, epoch_loss)

        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
#     print(net.weights[0].shape,net.weights[1].shape,net.biases[0].shape,net.biases[1].shape)
    dev_pred = net(dev_input)
    dev_rmse = optimizer.rmse(dev_target, dev_pred)
    
    train_pred = net(train_input)
    train_rmse = optimizer.rmse(train_target, train_pred)

    print('RMSE on train data: {:.5f}'.format(train_rmse))
    print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
    dev_pred = net(inputs)
    return dev_pred

def read_data():
    
    train = pd.read_csv("dataset/train.csv")
    dev = pd.read_csv("dataset/dev.csv")
    test_input = pd.read_csv("dataset/test.csv")
    
    train = train.to_numpy()
    train_input = train[:,1:].astype(float)
    train_target = train[:,0].astype(float).reshape(len(train),1)
    
    dev = dev.to_numpy()
    dev_input = dev[:,1:].astype(float)
    dev_target = dev[:,0].astype(float).reshape(len(dev),1)
    
    test_input = test_input.to_numpy()

    return train_input, train_target, dev_input, dev_target, test_input

def main():

    # These parameters should be fixed for Part 1
    max_epochs = 50
    batch_size = 48


    learning_rate = 0.001
    num_layers = 2
    num_units = 128
    lamda = 0.2 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    a = get_test_data_predictions(net, test_input)
    return a


if __name__ == '__main__':
    a = main()

