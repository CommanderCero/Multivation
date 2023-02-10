import torch

from stable_baselines3.common.buffers import ReplayBufferSamples
from abc import ABC, abstractmethod

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
    
class ExtrinsicRewardGenerator(RewardGenerator):
    def __init__(self, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return samples.rewards
    
    def from_config(yaml_node):
        return ExtrinsicRewardGenerator(
            reward_decay=yaml_node["reward_decay"],
            use_dones=yaml_node["use_dones"],
        )
    
class NegativeOneRewardGenerator(RewardGenerator):
    def __init__(self, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
    
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return torch.full(samples.rewards.shape, -1)
    
class CuriosityRewardGenerator(RewardGenerator):
    def __init__(self, learning_rate, reward_decay: float=0.99, use_dones: bool=True):
        super().__init__(reward_decay=reward_decay, use_dones=use_dones)
        
        self.embedding_net = torch.Sequential()
        self.forward_model = torch.Sequential()
        self.inverse_forward_model = torch.Sequential()
        
        self.optimizer = torch.optim.Adam([
            *self.forward_model.parameters(),
            *self.inverse_forward_model.parameters(),
            *self.embedding_net.parameters()
        ], lr=learning_rate, eps=1e-4)
        
    def generate_rewards(self, samples: ReplayBufferSamples) -> torch.Tensor:
        return torch.full(samples.rewards.shape, -1)