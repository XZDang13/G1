import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import MLPLayer

class EncoderNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dims:list[int]):
        super().__init__()
        
        self.layers = nn.ModuleList(self.init_layers(state_dim, hidden_dims))

    def init_layers(self, in_dim:int, hidden_dims:list[int]):
        layers = []
        dim = in_dim
        
        for hidden_dim in hidden_dims:
            mlp = MLPLayer(dim, hidden_dim, F.silu, True)
            dim = hidden_dim

            layers.append(mlp)

        self.dim = dim
        return layers

    def forward(self, x:torch.Tensor, aug:bool=False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = F.dropout(x, p=0.25, training=aug)

        return x