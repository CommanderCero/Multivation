import torch

from memory import ExperienceBatch

from abc import ABC, abstractmethod

class RewardGenerator(ABC):
    @abstractmethod
    def generate_rewards(self, samples: ExperienceBatch) -> torch.Tensor:
        pass
    
class ExtrinsicRewardGenerator(RewardGenerator):
    def __init__(self):
        pass
    
    def generate_rewards(self, samples: ExperienceBatch) -> torch.Tensor:
        return samples.rewards
    
class NegativeOneRewardGenerator(RewardGenerator):
    def __init__(self):
        pass
    
    def generate_rewards(self, samples: ExperienceBatch) -> torch.Tensor:
        return torch.full(samples.rewards.shape, -1)