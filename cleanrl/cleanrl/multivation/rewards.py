import gym

import torch
from stable_baselines3.common.buffers import ReplayBufferSamples

from models import Conv2DEmbedding, OneHotForwardModel, DiscreteActionPredictor

from abc import ABC, abstractmethod
from typing import Dict

class RewardGenerator(ABC):
    def __init__(self, reward_decay: float=0.99, use_dones: bool=True):
        self.reward_decay = reward_decay
        self.use_dones = use_dones
        
    def generate_data(self, samples: ReplayBufferSamples):
        dones = samples.dones if self.use_dones else torch.zeros(samples.dones.shape)
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
    def from_config(cls, yaml_node, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Box):
        return cls(
            reward_decay=yaml_node["reward_decay"],
            use_dones=yaml_node["use_dones"],
        )
    
class ExtrinsicRewardGenerator(RewardGenerator):
    def __init__(self, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return samples.rewards
    
class NegativeOneRewardGenerator(RewardGenerator):
    def __init__(self, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return torch.full(samples.rewards.shape, -1)
    
class CuriosityRewardGenerator(RewardGenerator):
    def __init__(self, state_shape, embedding_size, num_actions, learning_rate=0.0003, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
        
        self.embedding_net = Conv2DEmbedding(state_shape, embedding_size)
        self.forward_model = OneHotForwardModel(embedding_size, num_actions)
        self.inverse_forward_model = DiscreteActionPredictor(embedding_size, num_actions)
        
        self.optimizer = torch.optim.Adam([
            *self.forward_model.parameters(),
            *self.inverse_forward_model.parameters(),
            *self.embedding_net.parameters()
        ], lr=learning_rate, eps=1e-4)
    
    @torch.inference_mode()
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        state_embeddings = self.embedding_net(samples.observations)
        next_state_embeddings = self.embedding_net(samples.next_observations)
        next_state_predictions = self.forward_model(state_embeddings, samples.actions)
        
        rewards = torch.norm(next_state_predictions - next_state_embeddings, dim=1, p=2)
        return rewards
        
    def update(self, samples: ReplayBufferSamples) -> Dict[str, float]:
        """
        Updates the reward generator using the samples from the environment.
        Returns a dictionary of metrics that measure the training progress.
        """
        # Compute loss
        state_embeddings = self.embedding_net(samples.observations)
        next_state_embeddings = self.embedding_net(samples.next_observations)
        
        fm_loss = self.forward_model.compute_loss(state_embeddings, samples.actions, next_state_embeddings)
        inverse_fm_loss = self.inverse_forward_model.compute_loss(state_embeddings, next_state_embeddings, samples.actions)
        loss = 0.5 * fm_loss + 0.5 * inverse_fm_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "fm_loss": fm_loss.cpu().item(),
            "inverse_fm_loss": inverse_fm_loss.cpu().item()
        }
    
    @classmethod
    def from_config(cls, yaml_node, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Box):
        return cls(
            state_shape=obs_space.shape,
            num_actions=action_space.n,
            embedding_size=yaml_node["embedding_size"],
            learning_rate=yaml_node["learning_rate"],
            reward_decay=yaml_node["reward_decay"],
            use_dones=yaml_node["use_dones"],
        )