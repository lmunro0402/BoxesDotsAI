# Neural Network
#
# Author: Luke Munro

import numpy as np
import time


class Neuron:
    """Neuron for logistic regression. Given layer, index in layer, and size of data 
    it outputs a single value."""
    def __init__(self, sizeX, seed):
        np.random.seed(seed+30)
        # Initialize weights randomly with mean 0
        self.weights = 2*np.random.random((int(sizeX)))-1

    def assignW(self, weights):
        # print weights - self.weights
        self.weights = weights

    def getW(self):
        return self.weights
