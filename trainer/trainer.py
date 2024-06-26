import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
from tensorboardX import SummaryWriter
from memory_profiler import profile

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from utils.utils import *
from network.MLP import MLP
from algorithm.PPO import PPO
from trainer.player import Player


class Trainer():
    
    def __init__(self,
                 net: MLP,
                 algo,
                 eval_env: gym.Env,
                 vec_env: gym.vector.VectorEnv,
                 lr=1e-4,
                 exp_dir="./exp/LunarLander-v2",
                 all_kwargs=None):
        
        self.net = net
        self.algo = algo
        self.vec_env = vec_env
        self.optimizer = torch.optim.Adam(net.parameters(),
                                          lr=lr)
        self.exp_dir = exp_dir
        self.save_dir = os.path.join(exp_dir, str(int(time.time())))
        self.log_dir = os.path.join(self.save_dir, "log")
        self.ckpt_dir = os.path.join(self.save_dir, "ckpt")
        
        os.makedirs(self.save_dir)
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        self.log_writer = SummaryWriter(self.log_dir)
        
        cfg_path = os.path.join(self.save_dir, "config.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(all_kwargs, f)      
            
        self.player = Player() 
        self.eval_env = eval_env
    
    
    def save_model(self, ckpt_name):
        # save_dir = os.path.join(self.save_dir, "ckpt")
        # os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save({'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, save_path)


    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        
        
    def eval(self,
             global_step,
             evals_num=32):
        
        total_reward = self.player.play(self.net,
                                        self.eval_env,
                                        silence=True,
                                        evals_num=evals_num)
        
        self.log_writer.add_scalar("game total score", total_reward, global_step=global_step)
        
        
    def train(self,
              iters_n=10000,
              log_freq=20,
              save_freq=5000,
              eval_freq=2000,
              local_steps=8,
              epochs_n=20,
              batchs_n=8):
        
        # Pseudo code.
        net = self.net
        algo = self.algo
        vec_env = self.vec_env
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                      start_factor=1,
                                                      end_factor=0,
                                                      total_iters=iters_n)
        
        # Reset the environment.
        observation, info = vec_env.reset()
        envs_n = len(observation)
        
        mean_reward = 0
        net.train()
        
        # The main training process.
        for iter in tqdm(range(iters_n)):
            
            rewards = []
            values = []
            actions = []
            observations = []
            next_observations = []
            pis = []
            old_pis = []
            policies = []
            terminations = []
            
            with torch.no_grad():
                for t in range(local_steps):
                
                    observation = torch.tensor(observation, dtype=torch.float32)
                    value, policy = net(observation)
                    try:
                        action = sample_action(policy)
                    except:
                        import pdb; pdb.set_trace()
                    pi = policy[list(range(envs_n)), action]
                    old_pi = pi.clone()
                    
                    next_observation, reward, terminated, truncated, info = vec_env.step(action)
                    
                    rewards.append(torch.tensor(reward, dtype=torch.float32))
                    values.append(value)                            # torch.tensor
                    actions.append(action)
                    observations.append(observation)                # torch.tensor, need to be inputed in to network.
                    next_observations.append(torch.tensor(next_observation, dtype=torch.float32))
                    pis.append(pi)                          # torch.tensor
                    old_pis.append(old_pi)                  # torch.tensor
                    policies.append(policy)                 # torch.tensor
                    terminations.append(torch.tensor(terminated, dtype=torch.int))
                    
                    observation = next_observation
                    
                    if terminated.any() or truncated.any():         # FIXME: any() or all()?
                        break
            
            mean_reward = (mean_reward * 9 + torch.mean(torch.stack(rewards)).item()) / (9 + 1)   # Weighted sum.

            if len(values) > 1:
                info = algo.update(
                    net,
                    optimizer,
                    rewards=rewards,
                    values=values,
                    actions=actions,
                    observations=observations,
                    next_observations=next_observations,
                    pis=pis,
                    old_pis=old_pis,
                    policies=policies,
                    terminations=terminations,
                    epochs_n=epochs_n,
                    batchs_n=batchs_n
                )

                if iter % log_freq == 0:
                    # log_str = "Rewards: %f" % mean_reward
                    # print(log_str)
                    self.log_writer.add_scalar("mean reward", mean_reward, global_step=iter)
                    for key, loss in info.items():
                        self.log_writer.add_scalar(key, loss, global_step=iter)                   
            if iter % save_freq == 0 and iter != 0:
                self.save_model("%d.pt" % iter)
            
            if iter % eval_freq == 0 and iter != 0:
                self.eval(global_step=iter)
            
            scheduler.step()

if __name__ == '__main__':
    
    T = 8
    
    vec_env = gym.vector.make("LunarLander-v2", num_envs=4)
    net = MLP()
    algo = PPO(T=T, gamma=.99, lam=.95, epsilon=0.1, e_loss_weight=.01)
    trainer = Trainer(net, algo, lr=1e-4)
    
    trainer.train(vec_env, iters_n=10000, local_steps=T, log_freq=50)    
    
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    # for _ in range(1000):
    while True:
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        observation = torch.tensor(observation, dtype=torch.float32)
        value, policy = net(observation)
        action = sample_action(policy)
        observation, reward, terminated, truncated, info = env.step(action)
        # import pdb; pdb.set_trace()
        if terminated or truncated:
            observation, info = env.reset()

    env.close()    