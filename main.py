#!/usr/bin/env python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from Neuron import *
from training_methods import *
import matplotlib.pyplot as plt
import import_data as imp

#get salaries, other data
data,header = imp.import_baseball_data('Assignment_3_Hitters.csv')
def extract_salaries(input):
    salaries=[]
    output=[]
    #normalize as (y-min/max-min) everything but salary.
    for colIdx in range(len(input[0])):
        theseColValues = [row[colIdx] for row in input]
        for rowIdx,__ in enumerate(input):
            input[rowIdx][colIdx] = \
                (input[rowIdx][colIdx] - min(theseColValues))/(max(theseColValues) - min(theseColValues))

    for row in input:
        toutput = []
        for idx,col in enumerate(row):
            if header[idx] != "Salary":
                toutput.append(col)
            else:
                salaries.append(col)
        output.append(toutput)


    return output,salaries

data,salaries = extract_salaries(data)

#split into training and actual
xtrain, xtest, ytrain, ytest = train_test_split(data, salaries, test_size=0.2, random_state=10)

#pick a learning rate
def pick_a_learning_rate():
    plt.figure()
    for rate in [1,0.5,0.25,1e-1,0.5e-1,1e-2,1e-3]:
        num_epochs = 50
        NN = Neuron(len(xtrain[0]),\
                ActivationType=ReLU(lkg=0.05))
        [bgd_weight_history,trained_neuron] = \
            BGD(NN,num_epochs,rate,xtrain,ytrain)
        [itr,mse] = get_mse(NN,bgd_weight_history,xtrain,ytrain)
        plt.semilogy(itr,mse,label="Rate @ {}".format(rate))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("LearningRateSelection.png",dpi=1200)
    plt.close()

pick_a_learning_rate()

#assumptions
num_epochs = 20
learning_rate = 0.25

NN = Neuron(len(xtrain[0]),\
            ActivationType=ReLU(lkg=0.05),\
            RegularizationType=Ridge(lambda_cnst=0))
regularization_types = [Regularization(),
                        Ridge(lambda_cnst=0.01),
                        Ridge(lambda_cnst=10),
                        Lasso(lambda_cnst=0.01),
                        Lasso(lambda_cnst=10)]
weight_histories=[]
NNs=[]
types=["None","Ridge@0.01","Ridge@10","Lasso@0.01","Lasso@10"]

for regtype in regularization_types:
    NN = Neuron(len(xtrain[0]),\
            ActivationType=ReLU(lkg=0.05),\
            RegularizationType=regtype)
    NNs.append(NN)
    [weight_history,trained_neuron] = \
        BGD(NN,num_epochs,learning_rate,xtrain,ytrain)
    weight_histories.append(weight_history)

#generate plots
plt.figure()
colors = ["k","b","b","r","r'"]
for type,weights,regtype in zip(types,weight_histories,regularization_types):
    NN = Neuron(len(xtrain[0]),\
            ActivationType=ReLU(lkg=0.05),\
            RegularizationType=regtype)
    #we have histories per regularization type
    [itr,mse] = get_mse(NN,weights,xtrain,ytrain)
    p = plt.semilogy(itr,mse,label="Train, w/Reg Type = {}".format(type))
    [itr,mse] = get_mse(NN,weights,xtest,ytest)
    plt.semilogy(itr,mse,label="Test, w/Reg Type = {}".format(type),\
                color = p[-1].get_color(),linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc='upper left',fontsize=8)
plt.tight_layout()
plt.savefig("MSE.png",dpi=1200)
plt.legend(loc='upper right')
plt.ylim(top=10)
plt.savefig("MSE_limited.png",dpi=1200)
plt.close()

#generate table (print to csv)
#EXPORT FINAL WEIGHTS ONLY
import csv
with open('final_weights.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Weight"]+types)
    for idx in range(len(weight_histories[0])):
        values = [idx]
        for history in weight_histories:
            values.append(history[-1][idx])
        writer.writerow(values)
