from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RewardGeneratorConfig:
    reward_decay: float = 0.99
    use_dones: bool = True
    
@dataclass
class ExtrinsicRewardConfig(RewardGeneratorConfig):
    _target_: str = "rewards.ExtrinsicRewardGenerator"

@dataclass
class CuriosityRewardConfig(RewardGeneratorConfig):
    _target_: str = "rewards.CuriosityRewardGenerator"
    embedding_size: int = 128
    learning_rate: float = 3e-4

@dataclass
class RNDRewardConfig(RewardGeneratorConfig):
    _target_: str = "rewards.CuriosityRewardGenerator"
    embedding_size: int = 128
    learning_rate: float = 3e-4
    use_reward_norm: bool = True
    use_obs_norm: bool = True

@dataclass
class TrainConfig:
    experiment_name: str = "Experiment"
    env_id: str = "PongNoFrameskip-v4"
    seed: int = 1
    use_cuda: bool = True
    
    model_folder = "./models"
    save_model_frequency: int = int(1e5)
    
    total_timesteps: int = int(5e6)
    learning_starts: int = int(2e4)
    update_frequency: int = 4
    target_update_frequency: int = 8000
    switch_head_frequency: int = 10000
    
    batch_size: int = 64
    learning_rate: float = 3e-4
    memory_size: int= int(1e5)
    discount_factor: float = 0.99
    target_smoothing_coefficient: float = 1.0
    autotune_entropy: bool = True
    alpha: float = 0.2
    target_entropy_scale: float = 0.89
    share_body: bool = False
    
    reward_sources: Dict[str, RewardGeneratorConfig] = field(default_factory=lambda: {"extrinsic": ExtrinsicRewardConfig})
