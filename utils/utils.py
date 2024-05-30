import torch
import numpy as np

def sample_action(policy):
    
    action = torch.multinomial(policy, 1)
    action = action.numpy()
    
    if len(action.shape) == 2:
        action = action.reshape(-1,)
    else:
        action = action[0]
        
    return action


if __name__ == '__main__':
    
    policy = torch.tensor([0.2593, 0.2784, 0.2503, 0.2120], dtype=torch.float32)
    action = sample_action(policy)
    import pdb; pdb.set_trace()
    policy = torch.tensor(
        [[0.2428, 0.2631, 0.2404, 0.2537],
        [0.2428, 0.2631, 0.2404, 0.2537],
        [0.2428, 0.2631, 0.2404, 0.2537]], dtype=torch.float32
    )
    action = sample_action(policy)
    import pdb; pdb.set_trace()