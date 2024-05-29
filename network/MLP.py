import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    
    def __init__(self,
                 in_channels=8,
                 out_channels=4,
                 inner_channels=32,
                 layers_n=5):
        
        super(MLP, self).__init__()
        assert layers_n > 2
        self.start_layer = nn.Linear(in_channels, inner_channels)
        self.inner_layer_seq = nn.ModuleList([nn.Linear(inner_channels, inner_channels) for _ in range(layers_n-2)])
        self.policy_out_layer = nn.Linear(inner_channels, out_channels)
        self.value_out_layer = nn.Linear(inner_channels, 1)
        
    
    def forward(self, x):
        
        x = self.start_layer(x)
        x = F.relu(x)
        for layer in self.inner_layer_seq:
            x = layer(x)
            x = F.relu(x)
        policy = self.policy_out_layer(x)
        policy = F.softmax(policy)
        value = self.value_out_layer(x)
        
        return value, policy