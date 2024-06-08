import os
import torch
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO

 
if __name__ == '__main__':
    
    env = gym.make("LunarLander-v2", render_mode="human")
    player = Player()
    net = MLP()
    
    ckpt_path = "exp/LunarLander-v2/1717780164/ckpt/first.pt"
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt["model"])
    
    player.play(net, env)