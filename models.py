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
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        
        input_size = get_output_count(self.conv_net, input_shape)
        self.out_net = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
    def forward(self, X):
        X = self.conv_net(X)
        X = torch.flatten(X, start_dim=1)
        X = self.out_net(X)
        return X

class FMResidualBlock(nn.Module):
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size + num_actions, hidden_size)
        self.layer2 = nn.Linear(hidden_size + num_actions, hidden_size)
        
    def forward(self, x, actions):
        res = torch.cat([x, actions], dim=1)
        res = self.layer1(res)
        res = F.relu(res)
        
        res = torch.cat([x, actions], dim=1)
        res = self.layer1(res)
        return x + res
        

class OneHotForwardModelResiduals(nn.Module):
    '''
    Implements a forward model that predicts the next state using discrete actions as input.
    The actions are handled by using one-hot encoding.
    '''
    def __init__(self, embedding_size, hidden_size, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_size = embedding_size
        
        self.inp_net = nn.Sequential(
            nn.Linear(embedding_size + num_actions, hidden_size),
            nn.ReLU()
        )
        self.residuals = nn.ModuleList([
            FMResidualBlock(hidden_size, num_actions) for i in range(4)
        ])
        self.out_net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, embedding_size)
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
        
        X = self.inp_net(torch.cat([states, one_hot_actions], dim=1))
        for res_block in self.residuals:
            X = res_block(X, one_hot_actions)
        X = self.out_net(torch.cat([X, one_hot_actions], dim=1))
        return X
    
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
            nn.ReLU(),
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
            nn.Linear(self.embedding_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
    
    def forward(self, states, next_states):
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
    
def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class NHeadCritic(nn.Module):
    def __init__(self, num_heads, observation_shape, num_actions):
        super().__init__()
        self.num_heads = num_heads
        
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
        )

        with torch.inference_mode():
            output_dim = self.body(torch.zeros(1, *observation_shape)).shape[1]

        self.heads = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(output_dim, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, num_actions))
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x = self.body(x / 255.0)
        head_qvals = [head(x) for head in self.heads]
        return torch.stack(head_qvals, dim=0)


class NHeadActor(nn.Module):
    def __init__(self, num_heads, observation_shape, num_actions):
        super().__init__()
        self.num_heads = num_heads
        
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
        )

        with torch.inference_mode():
            output_dim = self.body(torch.zeros(1, *observation_shape)).shape[1]

        self.heads = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(output_dim, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, num_actions))
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x = self.body(x / 255.0)
        head_logits = [head(x) for head in self.heads]
        return torch.stack(head_logits, dim=0)

    def get_action(self, x):
        logits = self.forward(x)
        policy_dist = torch.distributions.Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs