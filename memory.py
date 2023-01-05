import gym
import numpy as np

from collections import namedtuple, deque

ExperienceBatch = namedtuple("ExperienceBatch", ["states", "actions", "next_states", "rewards", "dones"])

class ReplayMemory:
    def __init__(self, size: int, observation_space: gym.spaces, action_space: gym.spaces.Space):
        assert isinstance(observation_space, gym.spaces.Box), "ReplayMemory currently only supports gym.spaces.Box as observation_space"
        assert isinstance(action_space, gym.spaces.Discrete), "ReplayMemory currently only supports gym.spaces.Discrete as action_space"
        
        self.max_size = size
        self.size = 0
        self.current_pos = 0
        
        self.states = np.empty([self.max_size, *observation_space.shape], dtype=observation_space.dtype)
        self.actions = np.empty([self.max_size], dtype=action_space.dtype)
        self.next_states = np.empty([self.max_size, *observation_space.shape], dtype=observation_space.dtype)
        self.dones = np.empty([self.max_size], dtype=np.float32)
        self.rewards = np.empty([self.max_size], dtype=np.float32)
    
    def add(self, state, action, next_state, reward, done):
        self.states[self.current_pos] = state
        self.actions[self.current_pos] = action
        self.next_states[self.current_pos] = next_state
        self.dones[self.current_pos] = done
        self.rewards[self.current_pos] = reward
        
        self.current_pos = (self.current_pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size: int) -> ExperienceBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return ExperienceBatch(
            states = self.states[indices],
            actions = self.actions[indices],
            next_states = self.next_states[indices],
            dones = self.dones[indices],
            rewards = self.rewards[indices]
        )
        
    def __len__(self):
        return self.size


import gym
env = gym.make("CartPole-v1")
memory = ReplayMemory(1000000, env.observation_space, env.action_space)

state = env.reset()
for _ in range(10000):
    action = env.action_space.sample()
    new_state, reward, done, info = env.step(action)
    memory.add(state, action, new_state, reward, done)
    state = new_state
    
    if done:
        state = env.reset()