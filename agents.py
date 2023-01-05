import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import copy_parameters, sample_weighting_vector
from rewards import RewardGenerator
from memory import ReplayMemory

from typing import List, Callable

class DiscreteMultivationSAC:
    def __init__(self, 
            actor: nn.Module, 
            critic_template: Callable[[], nn.Module], 
            reward_sources: List[RewardGenerator],
            memory: ReplayMemory,
            reward_decay: float = 0.99,
        ):
        self.actor = actor
        self.local_critic_1 = critic_template()
        self.local_critic_2 = critic_template()
        self.target_critic_1 = critic_template()
        self.target_critic_2 = critic_template()
        
        self.reward_decay = reward_decay
        self.reward_sources = reward_sources
        assert len(self.reward_sources) == self.actor.num_heads, "The number of actor-heads do not match the number of reward generators."
        assert len(self.reward_sources) == self.local_critic_1.num_heads, "The number of critic-heads do not match the number of reward generators."
        
        self.memory = memory
        
        # Ensure local and target critics are the same
        copy_parameters(self.local_critic_1, self.target_critic_1)
        copy_parameters(self.local_critic_2, self.target_critic_2)
        
    def train(self,
            env: gym.Env,
            total_steps: int,
            initialisation_steps: int = 20000,
            update_interval: int = 100,
            batch_size: int = 64,
            update_steps: int = 4,
        ):
        assert isinstance(env.action_space, gym.spaces.Discrete), "DiscreteMultivationSAC only supports environments with a discrete action space."
        
        steps_taken = 0
        head_weightings = sample_weighting_vector(self.actor.num_heads)
        state = env.reset()
        while steps_taken < total_steps:
            # Sample a action
            if steps_taken <= initialisation_steps:
                action = env.action_space.sample()
            else:
                action = self.sample_actions(torch.Tensor([state]), head_weightings).item()
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            self.memory.add(state, action, next_state, reward, done)
            state = next_state
            
            if done:
                state = env.reset()
                head_weightings = sample_weighting_vector(self.actor.num_heads)
            
            steps_taken += 1
            
            # Learn
            if steps_taken % update_interval == 0:
                for i in range(update_steps):
                    self.learn(batch_size=batch_size)
                    self.learn()
            
    def learn(self, batch_size: int = 64):
        """
        Performs one update step. For this, we first sample from the ReplayMemory and then use the data to perform the following steps:
        1. Perform one gradient step on the local critics to better approximate the Q-function.
        2. Perform one gradient step on the policy to act better in the environment.
        3. Interpolate the target critics parameters towards the local critics using polyak averaging.
        """
        pass
    
    def sample_actions(self, states: torch.FloatTensor, head_weightings: List[float]) -> torch.LongTensor:
        with torch.no_grad():
            head_probs = self.actor.compute_action_probabilities(states)
            
        # Combine probabilities using the weighting
        for i, weighting in enumerate(head_weightings):
            head_probs[i] *= weighting
        combined_probs = head_probs.sum(axis=0)
        
        return torch.distributions.Categorical(probs=combined_probs).sample()