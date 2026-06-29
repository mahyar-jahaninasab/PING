import torch 
import torch.nn as nn
from typing import List, Optional, Tuple 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
from dotenv import load_dotenv
import json 
import os 

load_dotenv()
problem_path = os.getenv('PROBLEM')
with open(problem_path) as f:
    data = json.load(f)
GENERATOR = data["GENERATOR"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module = nn.Tanh,
        dropout: Optional[float] = None,
        use_batchnorm: bool = False, 
        last_activation: Optional[nn.Module] = None,
        parent = True
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        if last_activation is not None:
            layers.append(last_activation())
        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.parent = parent

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.parent:
            lb = x[:,0:1, :]  
            ub = x[:,-1:, :]  
            x = (x - lb) / (ub - lb + 1e-8)
                        
        x = self.net(x)

        if self.parent:
            raw_start = x[:, 0, :]
            raw_end   = x[:, -1, :]
            delta = raw_end - raw_start         
            delta_safe = torch.where(
                delta.abs() > 1e-8,
                delta,
                torch.ones_like(delta),
            )
            raw_start = raw_start.unsqueeze(1)       
            delta_safe= delta_safe.unsqueeze(1)
            tau =  (x - raw_start) / delta_safe   
            b = torch.tensor(GENERATOR["start"], dtype=x.dtype, device=x.device).view(1, 1, -1)  * (1 - tau) + torch.tensor(GENERATOR["goal"], dtype=x.dtype, device=x.device).view(1, 1, -1)  * tau  
            g = tau * (1 - tau)
            x = b + g * x 

        return x

class Maxout(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 2):

        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k            = k
        self.lin = nn.Linear(in_features, out_features * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc = self.lin(x)
        fc = fc.view(x.size(0), self.out_features, self.k)
        return fc.max(dim=2)[0]
    
def make_interval_counts(
    num_rows: int,
    bonds: List[int],
    include_edges: bool = True,
    dtype: torch.dtype = torch.long,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bset = sorted(set(int(x) for x in bonds))
    if include_edges:
        bset = sorted(set([0, num_rows] + bset))
    if any(x < 0 or x > num_rows for x in bset):
        raise ValueError("All bonds must be within [0, num_rows]")
    diffs = []
    for i in range(len(bset) - 1):
        start, end = bset[i], bset[i + 1]
        if end < start:
            raise ValueError("Bonds must be non-decreasing after sorting")
        diffs.append(end - start)
    return torch.tensor(diffs, dtype=dtype, device=device).view(-1, 1)




def forward_with_boundary_mapping(raw_output , x: torch.Tensor) -> torch.Tensor:
    
    start_raw = raw_output[0]   
    end_raw = raw_output[-1]    
    start_target = torch.tensor([0.0, 0.0], device=x.device)
    end_target = torch.tensor([8.0, 8.0], device=x.device)
    scaled_output = torch.zeros_like(raw_output)
    for dim in range(raw_output.shape[1]):
        raw_start = start_raw[dim]
        raw_end = end_raw[dim]
        target_start = start_target[dim]
        target_end = end_target[dim]
        if torch.abs(raw_end - raw_start) > 1e-8:
            a = (target_end - target_start) / (raw_end - raw_start)
            b = target_start - a * raw_start
            scaled_output[:, dim] = a * raw_output[:, dim] + b
        else:
            t_values = torch.linspace(0, 1, raw_output.shape[0], device=x.device)
            scaled_output[:, dim] = target_start + (target_end - target_start) * t_values
    return scaled_output
