import numpy as np
from Regularization import *
from Activation import *

class Neuron:
    def __init__(self,input_size,start_weight=None,
                    ActivationType=Activation(),
                    RegularizationType=Regularization()):
        self.input_size = input_size
        self.weight_size = input_size+1
        if start_weight is None:
            self.weights = np.zeros(1+input_size)
        else:
            raise ValueError("Need weight")
        self.weight_history = []

        self.activation = ActivationType
        self.regularization = RegularizationType

    def set_weights(self,weights):
        if(len(weights)!=self.weight_size):
            raise ValueError("Incorrect size for weights!")
        self.weights = weights

    def combine(self,inputs):
        return np.dot(self.weights[1:],inputs)+self.weights[0]

    def predict(self,inputs):
        if (len(inputs) != self.input_size):
            raise ValueError("Input != input size")
        else:
            return self.activation.value(self.combine(inputs))

    def deriv(self,x,err):
        #so Error = MSE + Regularization
        #MSE = 1/N * sum((yi - yi')^2), where yi' = y'(w,x) ie weights+inputs
        #weight = "w"
        #d(Error)/d(w) = d(MSE)/d(w) + d(R)/d(w)
        #d(MSE)/d(w) = 2/N * sum((y-y')*d(y')/d(w))
        #print(self.weights)
        dRdw = self.regularization.deriv(self.weights)
        #print(dRdw)
        dMSEdw = -2*(err)*self.activation.deriv(x,self.weights)
        dMSEdw[1:] *= x
        return dMSEdw + dRdw

    def get_mse(self,x,y):
        return ((y-self.predict(x))**2,self.regularization.value(self.weights))
