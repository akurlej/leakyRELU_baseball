import numpy as np

class Perceptron:
    def __init__(self,input_size,start_weight=None):
        self.input_size = input_size
        self.weight_size = input_size+1
        if start_weight is None:
            self.weights = np.zeros(1+input_size)
        else:
            raise ValueError("Need weight")
        self.weight_history = []

    def set_weights(self,weights):
        if(len(weights)!=self.weight_size):
            raise ValueError("Incorrect size for weights!")
        self.weights = weights

    def predict(self,inputs):
        if (len(inputs) != self.input_size):
            raise ValueError("Input != input size")
        else:
            inputs = inputs
            value = sum([w*x for w,x in zip(self.weights[1:],inputs)])+self.weights[0]
            return (1 if value>=0 else 0)

    def deriv(self,x,err):
        #err is given by (y-p(x))^2
        #px is w0 + wi*xi
        #so d((y-p(x))^2)/w0 = -2*(err)
        #so d((y-p(x))^2)/wi = -2*(err)*xi
        deriv = np.zeros(self.weight_size)
        deriv[0] = -2*(err)
        deriv[1:] = -2*(err)*x
        return deriv
