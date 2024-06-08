import os
import yaml
import torch
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO

 
if __name__ == '__main__':
    
    conf_path = "./config/LunarLander-v2.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)        
    
    net = MLP(**kwargs["net"])
    env = gym.make(kwargs["env"]["env_name"], render_mode="human") 
    player = Player()
    
    ckpt_path = "exp/LunarLander-v2/1717780164/ckpt/first.pt"
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt["model"])
    
    player.play(net, env)