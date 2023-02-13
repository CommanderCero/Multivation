import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def get_output_count(net: nn.Module, input_shape: tuple):
    with torch.no_grad():
        o = net(torch.zeros(1, *input_shape))
        return int(np.prod(o.size()))

class Conv2DEmbedding(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
        )
        
        input_size = get_output_count(self.conv_net, input_shape)
        self.out_net = nn.Sequential(
            nn.Linear(input_size, embedding_size)
        )
        
    def forward(self, X):
        X = X.float() / 255.0
        X = self.conv_net(X)
        X = torch.flatten(X, start_dim=1)
        X = self.out_net(X)
        return X

class OneHotForwardModel(nn.Module):
    '''
    Implements a forward model that predicts the next state using discrete actions as input.
    The actions are handled by using one-hot encoding.
    '''
    def __init__(self, embedding_size, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_size = embedding_size
        
        self.state_predictor_net = nn.Sequential(
            nn.Linear(self.embedding_size + self.num_actions, self.embedding_size * 2),
            nn.Linear(self.embedding_size * 2, self.embedding_size)
        )
    
    def forward(self, states, actions):
        '''
        Predicts the next state from the given (state, action) pairs.
        Assumptions:
            states: [batch_size, embedding_size]
            actions: [batch_size, 1]
        '''
        # [batch_size, 1, num_actions]
        one_hot_actions = F.one_hot(actions.long(), num_classes=self.num_actions)
        # [batch_size, num_actions]
        one_hot_actions = one_hot_actions.squeeze(1)
        
        # [batch_size, embedding_size + num_actions]
        X = torch.cat([states, one_hot_actions], dim=1)
        # [batch_size, embedding_size]
        state_predictions = self.state_predictor_net(X)
        return state_predictions
    
    def compute_loss(self, states, actions, next_states):
        '''
        Computes a loss indicating how good this model is in predicting the next_state from (state, action) pairs.
        The used loss is the Mean Squared Error between the real and predicted next_state.
        Assumptions:
            states: [batch_size, embedding_size]
            actions: [batch_size, 1]
            next_states: [batch_size, embedding_size]
            
        '''
        predicted_next_states = self.forward(states, actions)
        return F.mse_loss(predicted_next_states, next_states)
    
class DiscreteActionPredictor(nn.Module):
    def __init__(self, embedding_size, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_size = embedding_size
        
        self.logits_net = nn.Sequential(
            nn.Linear(self.embedding_size * 2, 256),
            nn.Linear(self.embedding_size * 2, self.num_actions)
        )
    
    def forward(self, states, next_states) -> torch.distributions.Distribution:
        '''
        Predicts the executed actions from the given (state, next_state) pairs.
        Returns logits for each possible action.
        Assumptions:
            states: [batch_size, embedding_size]
            next_states: [batch_size, embedding_size]
        '''
        # [batch_size, embedding_size * 2]
        X = torch.concat([states, next_states], dim=1)
        # [batch_size, num_actions]
        logits = self.logits_net(X)
        return logits
    
    def compute_loss(self, states, next_states, actions):
        '''
        Computes a loss indicating how good this model is in predicting the given actions from (state, next_state) pairs.
        The used loss is the cross-entropy loss between the real and predicted actions.
        Assumptions:
            states: [batch_size, embedding_size]
            next_states: [batch_size, embedding_size]
            actions: [batch_size, 1]
        '''
        logits = self.forward(states, next_states)
        return F.cross_entropy(logits, actions.long().flatten())