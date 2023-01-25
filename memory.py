import torch

import gym
import numpy as np

from collections import namedtuple

ExperienceBatch = namedtuple("ExperienceBatch", ["states", "actions", "next_states", "rewards", "dones"])

class ReplayMemory:
    def __init__(self, size: int, num_envs: int, observation_space: gym.spaces, action_space: gym.spaces.Space):
        assert isinstance(observation_space, gym.spaces.Box), "ReplayMemory currently only supports gym.spaces.Box as observation_space"
        assert isinstance(action_space, gym.spaces.Discrete), "ReplayMemory currently only supports gym.spaces.Discrete as action_space"
        
        self.max_size = size
        self.num_envs = num_envs
        self.size = 0
        self.current_pos = 0
        
        self.states = np.empty([self.max_size, num_envs, *observation_space.shape], dtype=observation_space.dtype)
        self.actions = np.empty([self.max_size, num_envs], dtype=action_space.dtype)
        self.next_states = np.empty([self.max_size, num_envs, *observation_space.shape], dtype=observation_space.dtype)
        self.dones = np.empty([self.max_size, num_envs], dtype=np.float32)
        self.rewards = np.empty([self.max_size, num_envs], dtype=np.float32)
    
    def add(self, states, actions, next_states, rewards, dones):
        self.states[self.current_pos] = states
        self.actions[self.current_pos] = actions
        self.next_states[self.current_pos] = next_states
        self.dones[self.current_pos] = dones
        self.rewards[self.current_pos] = rewards
        
        self.current_pos = (self.current_pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
            
    def sample(self, batch_size: int, device: torch.device) -> ExperienceBatch:
        env_ids = np.random.randint(0, self.num_envs, size=batch_size)
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return ExperienceBatch(
            states = torch.from_numpy(self.states[indices, env_ids]).to(device),
            actions = torch.from_numpy(self.actions[indices, env_ids]).to(device),
            next_states = torch.from_numpy(self.next_states[indices, env_ids]).to(device),
            dones = torch.from_numpy(self.dones[indices, env_ids]).to(device),
            rewards = torch.from_numpy(self.rewards[indices, env_ids]).to(device)
        )
        
    def __len__(self):
        return self.size