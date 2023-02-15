import torch
import gym
import numpy as np

from stable_baselines3.common.buffers import ReplayBufferSamples

class RolloutBuffer:
    def __init__(self,
            size: int,
            num_envs: int,
            observation_space: gym.spaces,
            action_space: gym.spaces.Space
        ):
        assert isinstance(observation_space, gym.spaces.Box), "ReplayMemory currently only supports gym.spaces.Box as observation_space"
        assert isinstance(action_space, gym.spaces.Discrete), "ReplayMemory currently only supports gym.spaces.Discrete as action_space"
        
        self.max_size = size
        self.num_envs = num_envs
        self.current_pos = 0
        
        self.states = np.empty([self.max_size, num_envs, *observation_space.shape], dtype=observation_space.dtype)
        self.actions = np.empty([self.max_size, num_envs], dtype=action_space.dtype)
        self.next_states = np.empty([self.max_size, num_envs, *observation_space.shape], dtype=observation_space.dtype)
        self.dones = np.empty([self.max_size, num_envs], dtype=np.float32)
        self.rewards = np.empty([self.max_size, num_envs], dtype=np.float32)
    
    def add(self, states, actions, next_states, rewards, dones):
        assert self.current_pos < self.max_size, "Rollout buffer overflow"
        
        self.states[self.current_pos] = states
        self.actions[self.current_pos] = actions
        self.next_states[self.current_pos] = next_states
        self.dones[self.current_pos] = dones
        self.rewards[self.current_pos] = rewards
        self.current_pos += 1
        
    def flush_data(self):
        assert self.is_full, "Cannot flush data, buffer is not full yet"
        self.current_pos = 0
        
        return ReplayBufferSamples(
            observations = torch.from_numpy(self.swap_and_flatten(self.states)),
            actions = torch.from_numpy(self.swap_and_flatten(self.actions)),
            next_observations = torch.from_numpy(self.swap_and_flatten(self.next_states)),
            dones = torch.from_numpy(self.swap_and_flatten(self.dones)),
            rewards = torch.from_numpy(self.swap_and_flatten(self.rewards))
        )
    
    @property
    def is_full(self):
        return self.current_pos == self.max_size
    
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])