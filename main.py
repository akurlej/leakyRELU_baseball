#!/usr/bin/env python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from Perceptron import *
from training_methods import *
import matplotlib.pyplot as plt

#import iris data + normalize it
iris = load_iris()
def normalize_iris_type(target,name):
    corrected_target = np.ones(len(target))
    for x in range(len(target)):
        if name[target[x]] != "setosa":
            corrected_target[x]=0
    return corrected_target

iris_data = iris.data
iris_data_name = iris.feature_names
iris_type_name = iris.target_names
iris_type = normalize_iris_type(iris.target,iris_type_name)

#split into training and actual
xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_type, test_size=0.2, random_state=10)

#assumptions:
#num_epochs=10 was too low for training BGD, upped to 20
#learning_rate @ unity for no real reason.
num_epochs = 20
learning_rate = 1

[bgd_weight_history,bgd_error_count,bgd_trained_perceptron] = \
    BGD(Perceptron(4),num_epochs,learning_rate,xtrain,ytrain)
print("BGD #Error = {}".format(bgd_error_count))

[sgd_weight_history,sgd_error_count,sgd_trained_perceptron] = \
    SGD(Perceptron(4),num_epochs,learning_rate,xtrain,ytrain)
print("SGD #Error = {}".format(sgd_error_count))

[mini_weight_history,mini_error_count,mini_trained_perceptron] = \
    MINIBATCH(Perceptron(4),num_epochs,learning_rate,12,xtrain,ytrain)
print("Minibatch #Error = {}".format(mini_error_count))

def get_mse(weight_history,X,Y):
    p = Perceptron(4)
    mse = []
    itr = []
    for idx,weights in enumerate(weight_history):
        p.set_weights(weights)
        this_mse = 0
        for x,y in zip(X,Y):
            px = p.predict(x)
            this_mse += (y-px)**2
        mse.append(this_mse/len(Y))
        itr.append(idx)
    return itr,mse

#generate plots
#BGD train & actual
plt.figure()
[itr,mse] = get_mse(bgd_weight_history,xtrain,ytrain)
plt.plot(itr,mse,"ro",label='Training')
[itr,mse] = get_mse(bgd_weight_history,xtest,ytest)
plt.plot(itr,mse,"b*",label='Test')
plt.xlabel("Training Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Training Epoch [BGD]")
plt.legend()
plt.savefig("MSE_BGD.png")

plt.figure()
[itr,mse] = get_mse(sgd_weight_history,xtrain,ytrain)
plt.plot(itr,mse,"ro",label='Training')
[itr,mse] = get_mse(sgd_weight_history,xtest,ytest)
plt.plot(itr,mse,"b*",label='Test')
plt.xlabel("Training Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Training Epoch [SGD]")
plt.legend()
plt.savefig("MSE_SGD.png")

plt.figure()
[itr,mse] = get_mse(mini_weight_history,xtrain,ytrain)
plt.plot(itr,mse,"ro",label='Training')
[itr,mse] = get_mse(mini_weight_history,xtest,ytest)
plt.plot(itr,mse,"b*",label='Test')
plt.xlabel("Training Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Training Epoch [Minibatch=12]")
plt.legend()
plt.savefig("MSE_Minibatch12.png")
