import gymnasium as gym
import numpy as np
from copy import deepcopy

# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#     import pdb; pdb.set_trace()
#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()

env = gym.make("LunarLander-v2")
observation, info = env.reset()

envs_num = 3
env_vec = gym.vector.SyncVectorEnv([lambda: deepcopy(env) for _ in range(envs_num)])
actions = np.array([1, 1, 1])
observations, rewards, termination, truncation, infos = env_vec.step(actions)
import pdb; pdb.set_trace()