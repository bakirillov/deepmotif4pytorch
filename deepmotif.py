import torch
import numpy as np
from torch import nn
from weblogolib import *
from copy import deepcopy
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class DeepMotif(nn.Module):
    
    def __init__(self, size, objective, reg):
        """
        Initialize DeepMotif class
        
        Parameters:
            size - the size of motif, (height, length)
            objective - the function to minimize
            reg - regularization parameter
        """
        super(DeepMotif, self).__init__()
        M = np.array(
            [1/size[1]]*np.product(size)
        ).reshape(size)
        self.size = size
        # Motif weights
        self.W = nn.Parameter(torch.from_numpy(M).type(torch.FloatTensor))
        self.O = objective
        self.R = reg
        
    def train(self, optimizer, epochs):
        """
        Run optimization for the best motif weights
        
        Parameters:
            optimizer - optimizer to use
            epochs - number of epochs
        """
        for a in np.arange(epochs):
            optimizer.zero_grad()
            loss = self.O(self.W) + self.R*(self.W.norm()**2)
            loss.backward()
            optimizer.step()
        #We can't have negative weights, so abs is needed,
        self.W.data = torch.clamp(torch.abs(self.W.data), 0, 1)
        
    def plot(self):
        """Plot the motif matrix"""
        plt.imshow(self.M)
        plt.yticks([])
        plt.xticks(np.arange(self.size[1]))
        plt.show()
        
    @classmethod
    def create(cls, size, objective, reg=0.01, epochs=100):
        """
        Create and find the motif
        
        Parameters:
            size - the size of motif, (height, length)
            objective - the function to minimize
            reg - regularization parameter
            epochs - number of epochs
        """
        m = cls(size, objective, reg)
        opt = Adam(m.parameters())
        m.train(opt, epochs)
        m.M = m.W.cpu().data.numpy()
        # Max-normalization
        maximum = np.max(m.M)
        sums = np.sum(m.M, 0)
        m.M[:,sums==0] = 0
        m.M = m.M/maximum + 0.01
        # Sum-normalization
        sums = np.sum(m.M, 0)
        m.M = np.divide(m.M, sums)
        return(m)        