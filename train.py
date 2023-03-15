# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random

from models import NHeadActor, NHeadCritic
from nhead_sac import NHeadSAC

import gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import hydra
from train_config import TrainConfig
from omegaconf import OmegaConf

import logging
logger = logging.getLogger(__name__)

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, shape=env.observation_space.shape, dtype=np.float32)
        
    def observation(self, observation):
        return observation.astype(np.float32) / 255.

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def enforce_deterministic_results(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def initialize_agent(cfg: TrainConfig, action_space, observation_space, device) -> NHeadSAC:
    reward_sources = [
        hydra.utils.instantiate(source, device, observation_space.shape, action_space.n)
        for _, source in cfg.reward_sources.items()
    ]
    num_heads = len(reward_sources)
    
    rb = ReplayBuffer(
        cfg.memory_size,
        observation_space,
        action_space,
        device,
        handle_timeout_termination=True,
    )
    
    actor = NHeadActor(num_heads, observation_space.shape, action_space.n, share_body=cfg.share_body)
    critic_template = lambda: NHeadCritic(num_heads, observation_space.shape, action_space.n, share_body=cfg.share_body)
    
    agent = NHeadSAC(
        actor=actor, 
        critic_template=critic_template, 
        reward_sources=reward_sources,
        memory=rb,
        num_actions=action_space.n,
        entropy_weight=cfg.alpha,
        autotune_entropy=cfg.autotune_entropy,
        target_entropy_scale=cfg.target_entropy_scale,
        polyak_weight=cfg.target_smoothing_coefficient,
        learning_rate=cfg.learning_rate,
        device=device
    )
    return agent

@hydra.main(version_base="1.1", config_path="config", config_name="extrinsic_curious_adventure")
def main(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    logger.info(f"Using {device}")
    
    # Setup tensorboard logging
    writer = SummaryWriter("./")
    writer.add_text("config", OmegaConf.to_yaml(cfg))
    writer.add_text("device", str(device))
    # Setup model saving
    model_folder = f"{writer.get_logdir()}/models"
    os.makedirs(model_folder)
    # Seeding
    enforce_deterministic_results(cfg.seed)
    
    # Initialize environment
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id, cfg.seed)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only a discrete action space is supported"
    
    # Initialize agent
    agent = initialize_agent(cfg, envs.single_action_space, envs.single_observation_space, device)
    
    # Run Training
    agent.train(
        envs=envs,
        total_timesteps=cfg.total_timesteps,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        update_frequency=cfg.update_frequency,
        target_network_frequency=cfg.target_update_frequency,
        switch_head_frequency=cfg.switch_head_frequency,
        save_model_frequency=cfg.save_model_frequency,
        writer=writer,
        model_folder=model_folder
    )
    
    # End
    envs.close()
    writer.close()

if __name__ == "__main__":
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name="default", node=TrainConfig)
    main()
