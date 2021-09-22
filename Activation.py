import numpy as np

class Activation:
    def __init__(self):
        pass

    def value(self,inputs):
        return np.where(inputs>=0,1,0)

    def deriv(self,inputs,weights):
        deriv = np.ones(len(weights))
        return deriv

class ReLU(Activation):
    """
    x for x>=0
    lkg*x for x<0
    """
    def __init__(self,lkg):
        self.leakage = lkg

    def value(self,inputs):
        return np.where(inputs >=0, inputs, inputs*self.leakage)

    def deriv(self,inputs,weights):
        """
        derivative is
        1 for x>0
        lkg for x<0
        assume lkg for 0 (since not differentiable)
        """
        h = np.dot(inputs,weights[1:])+weights[0]
        def func(input):
            return np.where(input>=0,1,self.leakage)

        deriv = np.zeros(len(weights))
        deriv[:] = func(h)
        return deriv
