import os
import torch
import numpy as np
import gymnasium as gym
from memory_profiler import profile

from algorithm.BasicAlgorithm import BasicAlgorithm


class PPO(BasicAlgorithm):
    
    def __init__(self,
                 T,
                 gamma,
                 lam,
                 epsilon,
                 v_loss_weight=1,
                 e_loss_weight=.01,
                 reward_scale=.01,
                 phi=None,
                 eta=None):
        
        super(PPO, self).__init__(
                 T,
                 gamma,
                 lam,
                 epsilon,
                 v_loss_weight=v_loss_weight,
                 e_loss_weight=e_loss_weight,
                 reward_scale=reward_scale,
                 phi=phi,
                 eta=eta
        )    
        

    def update(self,
               net,
               optimizer,
               rewards,
               values,
               actions,
               observations,
               next_observations,
               pis,
               old_pis,
               policies,
               terminations,
               epochs_n=10,
               batchs_n=8):
        
        gamma = self.gamma
        lam = self.lam
        epsilon = self.epsilon
        reward_scale = self.reward_scale

        T = len(values)
        S = observations[0].shape[1]
        A = policies[0].shape[1]
        B = observations[0].shape[0]        
        
        # 0. Step forward.
        last_value, last_policy = net(next_observations[-1])
        values.append(last_value.detach())
        
        # 1. Estimate the Advantage values.
        cur_values  = torch.stack(values[:-1])            # [T, B, 1]
        next_values = torch.stack(values[1:]) * (1 - torch.stack(terminations)[..., None])  # [T, B, 1]
        cur_rewards = torch.stack(rewards)           
        cur_rewards = cur_rewards[..., None]              # [T, B, 1]
        cur_rewards = cur_rewards * self.reward_scale
        
        cur_delta = cur_rewards + gamma * next_values - cur_values
        cur_delta = cur_delta[..., 0]                     # [T, B]
        
        cur_adv = torch.matmul(self.weight_mat_list[T], cur_delta)      # [T, B]
        cur_adv = cur_adv
        cur_target = cur_rewards + gamma * next_values
        cur_target = cur_target[..., 0]           # [T, B]   
        
        cur_old_pis = torch.stack(old_pis)        # [T, B]
        cur_observations = torch.stack(observations)
        
        actions = torch.stack([torch.tensor(action) for action in actions])
        actions = actions.reshape(-1,)
        
        cur_old_pis = cur_old_pis.reshape(-1,)           # [BT]
        cur_adv = cur_adv.reshape(-1,)                   # [BT]
        cur_target = cur_target.reshape(-1,)             # [BT]   
    
        for _ in range(epochs_n):
            indice = torch.randperm(B * T)
            for iter in range(batchs_n):
                batch_indices = indice[
                                int(iter * (T * B / batchs_n)): int((iter + 1) * (
                                T * B / batchs_n))]
                # 2. Compute Clip loss.
                batch_cur_values, batch_cur_policies = net(cur_observations.reshape(-1, S)[batch_indices])     # [batch_size, S]
                batch_cur_pis = torch.stack(
                    [batch_cur_policies[t, actions[batch_indices][t]] for t in range(batch_cur_policies.shape[0])]
                )
                batch_cur_values = batch_cur_values.reshape(-1,)
                batch_cur_r = batch_cur_pis / cur_old_pis[batch_indices] 
            
                clip_loss = -torch.mean(
                    torch.min(
                        batch_cur_r * cur_adv[batch_indices],
                        torch.clamp(batch_cur_r, 1-epsilon, 1+epsilon) * cur_adv[batch_indices]
                    )
                )
                
                # 3. Compute v-loss.
                v_loss = self.value_loss(cur_target[batch_indices], batch_cur_values)
                
                # 4. Compute e-loss.
                min_real = torch.finfo(batch_cur_policies.dtype).min
                e_loss = -torch.mean(torch.sum(torch.clamp(batch_cur_policies, min=min_real) * torch.log(batch_cur_policies), dim=-1))
                # import pdb; pdb.set_trace()
                
                # 5. Update.
                loss = clip_loss + self.v_loss_weight * v_loss - self.e_loss_weight * e_loss
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
            
        return {
            "clip_loss": clip_loss.detach().cpu().item(),
            "v_loss": v_loss.detach().cpu().item(),
            "e_loss": e_loss.detach().cpu().item()
        }