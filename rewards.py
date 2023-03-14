import gym
import numpy as np

import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBufferSamples
from stable_baselines3.common.running_mean_std import RunningMeanStd

from models import Conv2DEmbedding, OneHotForwardModel, DiscreteActionPredictor, OneHotForwardModelResiduals

from abc import ABC, abstractmethod
from typing import Dict

class RewardGenerator(torch.nn.Module):
    def __init__(self, device, state_shape, num_actions, reward_decay: float=0.99, use_dones: bool=True):
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
    
class ExtrinsicRewardGenerator(RewardGenerator):
    def __init__(self, device, state_shape, num_actions, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(device, state_shape, num_actions, reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return samples.rewards

class CuriosityRewardGenerator(RewardGenerator):
    def __init__(self,
             device, 
             state_shape,
             num_actions,
             embedding_size = 128,
             learning_rate=0.0003,
             reward_decay: float=0.99,
             use_dones: bool=True
         ):
        super().__init__(device, state_shape, num_actions, reward_decay=reward_decay, use_dones=use_dones)
        
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
    
class RNDRewardGenerator(RewardGenerator):
    def __init__(self,
            device,
            state_shape,
            num_actions,
            embedding_size=128,
            learning_rate=0.0003,
            reward_decay: float=0.99,
            use_dones: bool=True,
            use_reward_norm=True,
            use_obs_norm=True
        ):
        super().__init__(device, state_shape, num_actions, reward_decay=reward_decay, use_dones=use_dones)
        
        self.use_reward_norm = use_reward_norm
        self.use_obs_norm = use_obs_norm
        
        self.random_net = Conv2DEmbedding(state_shape, embedding_size, use_batch_norm=False).to(self.device)
        self.predictor_net = Conv2DEmbedding(state_shape, embedding_size, use_batch_norm=False).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=learning_rate, eps=1e-4)
        
        if self.use_reward_norm:
            self.reward_estimation = 0
            self.reward_moments = RunningMeanStd()
        if self.use_obs_norm:
            self.obs_moments = RunningMeanStd(shape=state_shape)
        
    @torch.inference_mode()
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        # Normalize observations
        next_observations = samples.next_observations
        if self.use_obs_norm:
            self.obs_moments.update(next_observations.cpu().numpy())
            next_observations = ((next_observations - torch.from_numpy(self.obs_moments.mean).to(self.device)) / torch.from_numpy(np.sqrt(self.obs_moments.var)).to(self.device)).clip(-5, 5)
        
        # Compute rewards
        random_embedding = self.random_net(samples.next_observations)
        embedding_prediction = self.predictor_net(samples.next_observations)
        rewards = torch.mean((random_embedding - embedding_prediction) ** 2, dim=-1)
        
        # Normalize rewards
        if self.use_reward_norm:
            reward_ts = np.empty(rewards.shape)
            for i, r in enumerate(rewards):
                self.reward_estimation = 0.99 * self.reward_estimation + r
                reward_ts[i] = self.reward_estimation
            self.reward_moments.update(reward_ts)
            rewards = rewards / np.sqrt(self.reward_moments.var)
        
        return rewards
        
    def update(self, samples: ReplayBufferSamples) -> Dict[str, float]:
        next_observations = samples.next_observations
        if self.use_obs_norm:
            next_observations = ((next_observations - torch.from_numpy(self.obs_moments.mean).to(self.device)) / torch.from_numpy(np.sqrt(self.obs_moments.var)).to(self.device)).clip(-5, 5)
        
        # Do not train random network
        with torch.no_grad():
            random_embedding = self.random_net(samples.next_observations)
        embedding_prediction = self.predictor_net(samples.next_observations)
        
        distillation_loss = F.mse_loss(embedding_prediction, random_embedding)
        self.optimizer.zero_grad()
        distillation_loss.backward()
        self.optimizer.step()
        
        return {
            "distillation_loss": distillation_loss
        }
