import os
import yaml
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO


if __name__ == '__main__':
    
    # Exp dir.
    exp_name = "LunarLander-v2"
    exp_dir = os.path.join("./exp", exp_name)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    # Training.
    net = MLP()
    vec_env = gym.vector.make("LunarLander-v2", num_envs=8)
    algo = PPO(T=32, gamma=.99, lam=.95, epsilon=0.1, e_loss_weight=.01)
    trainer = Trainer(net, algo, lr=1e-3, exp_dir=exp_dir)
    
    # Resuming.
    ckpt_path = "exp/LunarLander-v2/1717780164/ckpt/first.pt"
    trainer.load_model(ckpt_path=ckpt_path)
    
    trainer.train(vec_env=vec_env,
                  iters_n=10000,
                  log_freq=100,
                  local_steps=32,
                  steps_n=32)
    
    trainer.save_model("first.pt")