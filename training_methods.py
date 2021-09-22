from Perceptron import *
import numpy as np

def get_mse(n,weight_history,X,Y):
    mse = []
    itr = []
    for idx,weights in enumerate(weight_history):
        n.set_weights(weights)
        this_mse = 0
        for x,y in zip(X,Y):
            tmse,reg = n.get_mse(x,y)
            this_mse += (tmse/len(Y)) + reg
        mse.append(this_mse)
        itr.append(idx)
    return itr,mse

#BATCH GRADIENT DESCENT
def BGD(neuron,epochs,learning_rate,X,Y):
    weight_history = []
    error_count = []
    weights = neuron.weights
    for itr in range(epochs):
        weight_history.append(weights)
        deriv = np.zeros(neuron.weight_size)
        for x,y in zip(X,Y):
            fx = neuron.predict(x)
            deriv += neuron.deriv(x,y-fx)

        weights = weights - learning_rate*(deriv/len(Y))
        neuron.set_weights(weights)

    return weight_history,neuron

#STOCHASTIC GRADIENT DESCENT
def SGD(neuron,epochs,learning_rate,X,Y):
    weight_history = []
    error_count = []
    weights = neuron.weights
    for itr in range(epochs):
        weight_history.append(weights)

        deriv = np.zeros(neuron.weight_size)
        ctr = 0
        for x,y in zip(X,Y):
            ctr+=1
            fx = neuron.predict(x)
            deriv += neuron.deriv(x,y-fx)
            weights = weights - learning_rate*(deriv/ctr)
            neuron.set_weights(weights)

    return weight_history,neuron

#MINIBATCH GRADIENT DESCENT
def MINIBATCH(neuron,epochs,learning_rate,batch_sz,X,Y):
    weight_history = []
    error_count = []
    weights = neuron.weights
    for itr in range(epochs):
        weight_history.append(weights)

        deriv = np.zeros(neuron.weight_size)
        ctr = 0
        for x,y in zip(X,Y):
            ctr+=1
            fx = neuron.predict(x)
            deriv += neuron.deriv(x,y-fx)
            if ctr % batch_sz == 0:
                weights = weights - learning_rate*(deriv/ctr)
                neuron.set_weights(weights)

    return weight_history,neuron
