import torch
import numpy as np
from torch import nn
from weblogolib import *
from copy import deepcopy
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from weblogolib.colorscheme import nucleotide, taylor


class DeepMotif(nn.Module):
    
    def __init__(self, size, objective, reg, cuda=False):
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
        self.cuda = cuda and torch.cuda.is_available()
        self.W = nn.Parameter(
            torch.from_numpy(
                M
            ).type(torch.FloatTensor).cuda() if self.cuda else torch.from_numpy(
                M
            ).type(torch.FloatTensor)
        )
        self.O = objective
        self.R = reg
        # Weblogo alphabets
        self.n_alph = "AGCT"
        self.aa_alph = "ARNDCGQEHILKMFPSTWYV"
        
    def fit(self, optimizer, epochs):
        """
        Run optimization for the best motif weights
        
        Parameters:
            optimizer - optimizer to use
            epochs - number of epochs
        """
        for a in np.arange(epochs):
            optimizer.zero_grad()
            loss = self.O(self.W) + self.R*(self.W.norm()**2)
            loss = loss.cuda() if self.cuda else loss
            loss.backward()
            optimizer.step()
        #We can't have negative weights, so abs is needed,
        self.W.data = torch.clamp(self.W.data, 0, 1)
        
    def heatmap(self):
        """Plot the motif matrix as a heatmap"""
        plt.imshow(self.M)
        plt.yticks([])
        plt.xticks(np.arange(self.size[1]))
        plt.show()
        
    def logo(self, out_fn, logo_options=None):
        """
        Draw logo.png
        
        Parameters:
            out_fn - name of output file
            logo options - custom logo options
        """
        alphabet = self.n_alph if self.size[0] == 4 else self.aa_alph
        options = LogoOptions()
        if logo_options:
            options = logo_options
        else:
            options.title = "DeepMotif logo"
            options.color_scheme = nucleotide if self.size[0] == 4 else taylor
        data = LogoData.from_counts(alphabet, self.M.T)
        frm = LogoFormat(data, options)
        with open(out_fn, "wb") as oh:
            oh.write(png_formatter(data, frm))
        
    @classmethod
    def create(cls, size, objective, reg=0.01, epochs=100, cuda=False):
        """
        Create and find the motif
        
        Parameters:
            size - the size of motif, (height, length)
            objective - the function to minimize
            reg - regularization parameter
            epochs - number of epochs
        """
        m = cls(size, objective, reg, cuda)
        m.train()
        opt = Adam(m.parameters())
        m.fit(opt, epochs)
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