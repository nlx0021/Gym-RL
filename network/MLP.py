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
        x = F.tanh(x)
        for layer in self.inner_layer_seq:
            x = layer(x)
            x = F.tanh(x)
        policy = self.policy_out_layer(x)
        policy = F.softmax(policy, dim=1)
        value = self.value_out_layer(x)
        
        return value, policy
    
    

if __name__ == '__main__':
    
    observation = np.array([ 0.0047967 ,  1.41536   ,  0.48583737,  0.19731522, -0.00555138,
       -0.11004938,  0.        ,  0.        ],)
    observation = torch.tensor(observation, dtype=torch.float32)   # (8,)
    
    net = MLP()
    value, policy = net(observation)
    import pdb; pdb.set_trace()
    observation = observation[None, ...].repeat_interleave(3, dim=0)
    value, policy = net(observation)
    import pdb; pdb.set_trace()