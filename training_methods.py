from Perceptron import *
import numpy as np

#BATCH GRADIENT DESCENT
def BGD(perceptron,epochs,learning_rate,X,Y):
    weight_history = []
    error_count = []
    weights = perceptron.weights
    for itr in range(epochs):
        weight_history.append(weights)

        thisError = 0
        deriv = np.zeros(perceptron.weight_size)
        for x,y in zip(X,Y):
            fx = perceptron.predict(x)
            if (y != fx):
                thisError += 1
            deriv += perceptron.deriv(x,y-fx)

        weights = weights - learning_rate*(deriv/len(Y))
        error_count.append(thisError)
        perceptron.set_weights(weights)

    return weight_history,error_count,perceptron

#STOCHASTIC GRADIENT DESCENT
def SGD(perceptron,epochs,learning_rate,X,Y):
    weight_history = []
    error_count = []
    weights = perceptron.weights
    for itr in range(epochs):
        weight_history.append(weights)

        thisError = 0
        deriv = np.zeros(perceptron.weight_size)
        ctr = 0
        for x,y in zip(X,Y):
            ctr+=1
            fx = perceptron.predict(x)
            if (y != fx):
                thisError += 1
            deriv += perceptron.deriv(x,y-fx)
            weights = weights - learning_rate*(deriv/ctr)
            perceptron.set_weights(weights)

        error_count.append(thisError)
    return weight_history,error_count,perceptron

#MINIBATCH GRADIENT DESCENT
def MINIBATCH(perceptron,epochs,learning_rate,batch_sz,X,Y):
    weight_history = []
    error_count = []
    weights = perceptron.weights
    for itr in range(epochs):
        weight_history.append(weights)

        thisError = 0
        deriv = np.zeros(perceptron.weight_size)
        ctr = 0
        for x,y in zip(X,Y):
            ctr+=1
            fx = perceptron.predict(x)
            if (y != fx):
                thisError += 1
            deriv += perceptron.deriv(x,y-fx)
            if ctr % batch_sz == 0:
                weights = weights - learning_rate*(deriv/ctr)
                perceptron.set_weights(weights)

        error_count.append(thisError)
    return weight_history,error_count,perceptron
