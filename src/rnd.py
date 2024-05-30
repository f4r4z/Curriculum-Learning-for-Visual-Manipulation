import numpy as np
import gymnasium as gym
from torch import nn
import torch as th

class RNDNetworkLowDim(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(RNDNetworkLowDim, self).__init__()
        self.linear  = nn.Linear(input_dim, output_dim)
        self.double()
    
    def forward(self, x):
        return self.linear(x)