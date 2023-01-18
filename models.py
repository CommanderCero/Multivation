import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import create_feedforward, create_conv

from typing import List
from abc import ABC, abstractmethod

def get_output_count(net: nn.Module, input_shape: tuple) -> int:
    with torch.no_grad():
        o = net(torch.zeros(1, *input_shape))
        return int(np.prod(o.size()))

class Conv2DEmbedding(nn.Module):
    def __init__(self, input_shape, embedding_size, conv_channels=[32, 32, 32, 32], linear_sizes=[], activation=nn.ReLU):
        super().__init__()
        
        self.conv_net = create_conv([input_shape[0], *conv_channels], activation=nn.ReLU)
        input_size = get_output_count(self.conv_net, input_shape)
        self.out_net = create_feedforward([input_size, *linear_sizes, embedding_size], activation=nn.ReLU)
        
    def forward(self, X):
        X = self.conv_net(X)
        X = torch.flatten(X, start_dim=1)
        X = self.out_net(X)
        return X

class NHeadCritic(nn.Module):
    def __init__(self, body: nn.Module, heads: List[nn.Module]):
        super().__init__()
        
        self.body = body
        self.heads = nn.ModuleList(heads)
        
    def forward(self, states):
        embedded_states = self.body(states)
        q_values = [head(embedded_states) for head in self.heads]
        q_values = torch.stack(q_values, dim=0)
        return q_values
    
    def predict_head(self, states: torch.FloatTensor, head_index: int) -> torch.FloatTensor:
        embedded_states = self.body(states)
        q_values = self.heads[head_index](embedded_states)
        return q_values
    
    @property
    def num_heads(self):
        return len(self.heads)
    
    @staticmethod
    def create_pure_feedforward(body_layers, num_heads, head_layers, activation=nn.ReLU):
        body = create_feedforward(body_layers, activation=activation)
        heads = [create_feedforward(head_layers, activation=activation) for _ in range(num_heads)]
        return NHeadCritic(body, heads)
    
    @staticmethod
    def create_conv(input_shape, embedding_size, conv_channels, body_layers, num_heads, head_layers, activation=nn.ReLU):
        body = Conv2DEmbedding(input_shape, embedding_size, conv_channels, body_layers)
        heads = [create_feedforward([embedding_size, *head_layers], activation=activation) for _ in range(num_heads)]
        return NHeadCritic(body, heads)
    
class NHeadActor(nn.Module):
    def __init__(self, body: nn.Module, heads: List[nn.Module]):
        super().__init__()
        
        self.body = body
        self.heads = nn.ModuleList(heads)
        
    def forward(self, states):
        embedded_states = self.body(states)
        logits = [head(embedded_states) for head in self.heads]
        logits = torch.stack(logits, dim=0)
        return logits
    
    def predict_head(self, states: torch.FloatTensor, head_index: int) -> torch.FloatTensor:
        embedded_states = self.body(states)
        logits = self.heads[head_index](embedded_states)
        return logits
    
    def get_action_dist(self, states) -> torch.distributions.Categorical:
        logits = self.forward(states)
        return torch.distributions.Categorical(logits=logits)
    
    @property
    def num_heads(self):
        return len(self.heads)
        
    @staticmethod
    def create_pure_feedforward(body_layers, num_heads, head_layers, activation=nn.ReLU):
        body = create_feedforward(body_layers, activation=activation)
        heads = [create_feedforward(head_layers, activation=activation) for _ in range(num_heads)]
        return NHeadActor(body, heads)
    
    @staticmethod
    def create_conv(input_shape, embedding_size, conv_channels, body_layers, num_heads, head_layers, activation=nn.ReLU):
        body = Conv2DEmbedding(input_shape, embedding_size, conv_channels, body_layers)
        heads = [create_feedforward([embedding_size, *head_layers], activation=activation) for _ in range(num_heads)]
        return NHeadActor(body, heads)
    