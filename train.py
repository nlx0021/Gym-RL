import os
import yaml
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO


if __name__ == '__main__':
    
    conf_path = "./config/CartPole-v1.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)        
    
    exp_dir = kwargs["trainer"]["exp_dir"]
    net = MLP(**kwargs["net"])
    vec_env = gym.vector.make(kwargs["env"]["env_name"], num_envs=kwargs["env"]["num_envs"])
    algo = PPO(**kwargs["algo"])
    trainer = Trainer(net, algo, vec_env, all_kwargs=kwargs, **kwargs["trainer"])
    
    # Resuming.
    ckpt_path = kwargs["resume"]["ckpt_path"]
    if ckpt_path:
        trainer.load_model(ckpt_path=ckpt_path) 
        
    # Training.
    trainer.train(**kwargs["train"])   
    trainer.save_model("Final.pt")