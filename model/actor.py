import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, DeterminicHead, GuassianHead
from RLAlg.distribution import TruncatedNormal

class ActorFixNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)
        self.max_action = max_action
        self.policy = DeterminicHead(dim, action_dim, max_action)
        self.std_value = 1.0

    def forward(self, x:torch.Tensor, action:torch.Tensor=None):
        x = self.layers(x)
        
        mean = self.policy(x)
        std = torch.ones_like(mean) * self.std_value
        pi = TruncatedNormal(mean, std, -self.max_action, self.max_action)

        if action is None:
            action = pi.rsample()

        log_prob = pi.log_prob(action).sum(axis=-1)

        
        return pi, action, log_prob
    
class ActorLearnNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)
        self.max_action = max_action
        self.policy = GuassianHead(dim, action_dim, max_action)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None=None) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        pi, action, log_prob = self.policy(x, action)

        return pi, action, log_prob