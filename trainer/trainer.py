import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from utils.utils import *
from network.MLP import MLP


class Trainer():
    
    def __init__(self):
        pass
    
    def train(self,
              net: nn.Module,
              env: gym.Env,
              iters_n=10000):
        
        # Pseudo code.
        
        episode = []
        net.train()
        # Reset the environment.
        observation, info = env.reset()
        
        # The main training process.
        for iter in tqdm(range(iters_n)):
            
            observation = torch.tensor(observation, dtype=torch.float32)
            value, policy = net(observation)
            action = sample_action(policy)
            observation_prime, reward, terminated, truncated, info = env.step(action)
            
            episode.append((observation, action, reward))
            
            # Here Algorithm updates the network parameters.
            
            observation = observation_prime
            
            if terminated or truncated:
                observation, info = env.reset()
            
            

if __name__ == '__main__':
    
    trainer = Trainer()
    env = gym.make("LunarLander-v2")
    net = MLP()
    
    trainer.train(net, env, iters_n=10000)