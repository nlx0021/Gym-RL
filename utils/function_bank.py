import torch
import numpy as np

def phi_exp_factory(p=1, q=1):
    assert p % 2 == 1 & q % 2 == 1
    return lambda x: torch.exp(torch.pow(torch.abs(x), p/q) * (torch.tensor(x>0, dtype=torch.float32) * 2 - 1))

def phi_poly_factory(p=1, q=1):
    assert p % 2 == 1 & q % 2 == 1
    return lambda x: torch.pow(torch.abs(x), p/q) * (torch.tensor(x>0, dtype=torch.float32) * 2 - 1)

def phi_log_factory(p=1, q=1):
    assert p % 2 == 1 & q % 2 == 1
    phi_poly = phi_poly_factory(p, q)
    return lambda x: torch.log(phi_poly(x))

def get_phi(name="exp", p=1, q=1):
    
    if name == "exp":
        return phi_exp_factory(p, q)
    
    elif name == "poly":
        return phi_poly_factory(p, q)
    
    elif name == "log":
        return phi_log_factory(p, q)