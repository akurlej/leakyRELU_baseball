import numpy as np

class Regularization:
    def __init__(self,lambda_cnst=0.0):
        self.lambda_cnst = lambda_cnst

    def value(self,weights):
        return np.sum(np.zeros(len(weights)))

    def deriv(self,weights):
        return np.zeros(len(weights))

class Lasso(Regularization):
    """
        regularization = lambda*sum(abs(weight))
    """
    def __init__(self,lambda_cnst=0):
        super().__init__(lambda_cnst)

    def value(self,weights):
        return (self.lambda_cnst)*np.sum(np.abs(weights))

    def deriv(self,weights):
        return (self.lambda_cnst)*np.sign(weights)

class Ridge(Regularization):
    """
        regularization = lambda/2*sum(weight^2)
    """
    def __init__(self,lambda_cnst=0):
        super().__init__(lambda_cnst)

    def value(self,weights):
        return (self.lambda_cnst/2)*np.sum(np.dot(weights,weights))

    def deriv(self,weights):
        return (self.lambda_cnst)*weights
