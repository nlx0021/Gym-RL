import os
import numpy as np
import gymnasium as gym

from trainer.trainer import Trainer
from trainer.player import Player
from network.MLP import MLP
from algorithm.PPO import PPO


if __name__ == '__main__':
    
    # Components.
    env = gym.make("LunarLander-v2")
    trainer = Trainer()
    net = MLP()
    algo = PPO()
    
    # Train.
    trainer.train(net, env, iters_n=1000)