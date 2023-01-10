import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import create_feedforward

from typing import List
from abc import ABC, abstractmethod

class NHeadCritic(nn.Module):
    def __init__(self, body: nn.Module, heads: List[nn.Module]):
        super().__init__()
        
        self.body = body
        self.heads = heads
        
    def forward(self, states):
        embedded_states = self.body(states)
        q_values = [head(embedded_states) for head in self.heads]
        q_values = torch.stack(q_values, dim=0)
        return q_values
    
    @property
    def num_heads(self):
        return len(self.heads)
    
    @staticmethod
    def create_pure_feedforward(body_layers, num_heads, head_layers, activation=nn.ReLU):
        body = create_feedforward(body_layers, activation=activation)
        heads = [create_feedforward(head_layers, activation=activation) for _ in range(num_heads)]
        return NHeadCritic(body, heads)
    
class NHeadActor(nn.Module):
    def __init__(self, body: nn.Module, heads: List[nn.Module]):
        super().__init__()
        
        self.body = body
        self.heads = heads
        
    def forward(self, states):
        embedded_states = self.body(states)
        logits = [head(embedded_states) for head in self.heads]
        logits = torch.stack(logits, dim=0)
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
    