import os
import torch
import numpy as np
import gymnasium as gym


class BasicAlgorithm():
    
    def __init__(self,
                 T,
                 gamma,
                 lam,
                 epsilon,
                 v_loss_weight=1,
                 e_loss_weight=.01):
        
        self.T = T
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.value_loss = torch.nn.MSELoss()
        self.v_loss_weight = v_loss_weight
        self.e_loss_weight = e_loss_weight
        
        self.weight_mat_list = [
            torch.tensor(
                [
                    [0 if i < j else (gamma * lam) ** (i-j) for i in range(t-1)] for j in range(t-1)
                ]
            ) for t in range(1, T+1)
        ]
        
    
    def update():
        
        pass