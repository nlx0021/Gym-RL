import torch
import numpy as np
import gymnasium as gym

from utils.utils import *

class Player():
    
    def __init__(self):
        pass
    
    def play(self,
              net,
              env):
        
        observation, info = env.reset()        
        game_ct = 0
        total_reward = 0
        while True:
            # action = env.action_space.sample()  # agent policy that uses the observation and info
            observation = torch.tensor(observation, dtype=torch.float32)
            value, policy = net(observation)
            action = sample_action(policy)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # import pdb; pdb.set_trace()
            if terminated or truncated:
                observation, info = env.reset()        
                print("Over: %d. The reward is %f" % (game_ct, total_reward))
                game_ct += 1
                total_reward = 0