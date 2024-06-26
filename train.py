import os
import yaml
import torch
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO
from algorithm.phi_update import PhiUpdate
from utils.function_bank import *

ALGO = {
    "PPO": PPO,
    "phi_update": PhiUpdate
}

if __name__ == '__main__':
    
    conf_path = "./config/LunarLander-v2-phi.yaml"
    
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)       
         
    torch.set_num_threads(kwargs["world"]["threads_num"])
    
    exp_dir = kwargs["trainer"]["exp_dir"]
    net = MLP(**kwargs["net"])
    vec_env = gym.vector.make(kwargs["env"]["env_name"], num_envs=kwargs["env"]["num_envs"])
    env = gym.make(kwargs["env"]["env_name"])
    phi = get_phi(**kwargs["phi"])
    algo = ALGO[kwargs["alg"]](phi=phi, **kwargs["algo"])
    trainer = Trainer(net, algo, env, vec_env, all_kwargs=kwargs, **kwargs["trainer"])
    
    # Resuming.
    ckpt_path = kwargs["resume"]["ckpt_path"]
    if ckpt_path:
        trainer.load_model(ckpt_path=ckpt_path) 
        
    # Training.
    trainer.train(**kwargs["train"])   
    trainer.save_model("Final.pt")