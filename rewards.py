import gym
import numpy as np

import torch
from stable_baselines3.common.buffers import ReplayBufferSamples
from stable_baselines3.common.running_mean_std import RunningMeanStd

from models import Conv2DEmbedding, OneHotForwardModel, DiscreteActionPredictor, OneHotForwardModelResiduals

from abc import ABC, abstractmethod
from typing import Dict

class RewardGenerator(torch.nn.Module):
    def __init__(self, device, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__()
        
        self.reward_decay = reward_decay
        self.use_dones = use_dones
        self.device = device
        
    def generate_data(self, samples: ReplayBufferSamples):
        dones = samples.dones if self.use_dones else torch.zeros(samples.dones.shape, device=self.device)
        rewards = self.generate_rewards(samples)
        return rewards, dones, self.reward_decay
    
    @abstractmethod
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        pass
    
    def update(self, samples: ReplayBufferSamples) -> Dict[str, float]:
        """
        Updates the reward generator using the samples from the environment.
        Returns a dictionary of metrics that measure the training progress.
        """
        return {}
    
    @classmethod
    def from_config(cls, yaml_node, device, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Box):
        return cls(
            device,
            reward_decay=yaml_node["reward_decay"],
            use_dones=yaml_node["use_dones"],
        )
    
class ExtrinsicRewardGenerator(RewardGenerator):
    def __init__(self, device, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(device, reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return samples.rewards

class CuriosityRewardGenerator(RewardGenerator):
    def __init__(self, device, state_shape, embedding_size, num_actions, learning_rate=0.0003, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(device, reward_decay=reward_decay, use_dones=use_dones)
        
        self.embedding_net = Conv2DEmbedding(state_shape, embedding_size).to(self.device)
        self.forward_model = OneHotForwardModelResiduals(embedding_size, 512, num_actions).to(self.device)
        self.inverse_forward_model = DiscreteActionPredictor(embedding_size, num_actions).to(self.device)
        
        self.reward_estimation = 0
        self.reward_moments = RunningMeanStd()
        
        self.optimizer = torch.optim.Adam([
            *self.forward_model.parameters(),
            *self.inverse_forward_model.parameters(),
            *self.embedding_net.parameters()
        ], lr=learning_rate, eps=1e-4)
    
    @torch.inference_mode()
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        # Compute reward
        state_embeddings = self.embedding_net(samples.observations)
        next_state_embeddings = self.embedding_net(samples.next_observations)
        next_state_predictions = self.forward_model(state_embeddings, samples.actions)
        rewards = torch.mean((next_state_predictions - next_state_embeddings) ** 2, dim=-1)
        
        # Normalize
        reward_ts = np.empty(rewards.shape)
        for i, r in enumerate(rewards):
            self.reward_estimation = 0.99 * self.reward_estimation + r
            reward_ts[i] = self.reward_estimation
        self.reward_moments.update(reward_ts)
        
        return rewards / np.sqrt(self.reward_moments.var)
        
    def update(self, samples: ReplayBufferSamples) -> Dict[str, float]:
        """
        Updates the reward generator using the samples from the environment.
        Returns a dictionary of metrics that measure the training progress.
        """
        # Compute loss
        state_embeddings = self.embedding_net(samples.observations)
        next_state_embeddings = self.embedding_net(samples.next_observations)
        
        fm_loss = self.forward_model.compute_loss(state_embeddings.detach(), samples.actions, next_state_embeddings.detach())
        inverse_fm_loss = self.inverse_forward_model.compute_loss(state_embeddings, next_state_embeddings, samples.actions)
        loss = inverse_fm_loss + fm_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "fm_loss": fm_loss.cpu().item(),
            "inverse_fm_loss": inverse_fm_loss.cpu().item()
        }
    
    @classmethod
    def from_config(cls, yaml_node, device, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Box):
        return cls(
            device,
            state_shape=obs_space.shape,
            num_actions=action_space.n,
            embedding_size=yaml_node["embedding_size"],
            learning_rate=yaml_node["learning_rate"],
            reward_decay=yaml_node["reward_decay"],
            use_dones=yaml_node["use_dones"],
        )