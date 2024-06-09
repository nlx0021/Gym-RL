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
    
    conf_path = "exp\\CartPole-v1\\1717936955\\config.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)        
    
    net = MLP(**kwargs["net"])
    net = net.eval()
    env = gym.make(kwargs["env"]["env_name"], render_mode="human") 
    player = Player()
    
    ckpt_path = "exp\\CartPole-v1\\1717936955\\ckpt\\Final.pt"
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt["model"])
    
    player.play(net, env)