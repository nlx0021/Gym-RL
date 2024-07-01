import os
import torch
import numpy as np
import gymnasium as gym

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from utils.function_bank import *


class BasicAlgorithm():
    
    def __init__(self,
                 T,
                 gamma,
                 lam,
                 epsilon,
                 v_loss_weight=1,
                 e_loss_weight=.01,
                 reward_scale=.01,
                 phi="idendity",
                 eta=1):
        
        self.T = T
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.value_loss = torch.nn.MSELoss()
        self.v_loss_weight = v_loss_weight
        self.e_loss_weight = e_loss_weight
        self.eta = eta
        self.phi = phi
        self.reward_scale = reward_scale
        
        self.weight_mat_list = [
            torch.tensor(
                [
                    [0 if i < j else (gamma * lam) ** (i-j) for i in range(t-1)] for j in range(t-1)
                ]
            ) for t in range(1, T+2)
        ]
        
    
    def update():
        
        pass