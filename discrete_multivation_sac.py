import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_utils import copy_parameters, sample_weighting_vector
from rewards import RewardGenerator
from memory import ReplayMemory, ExperienceBatch
from models import NHeadActor, NHeadCritic

from typing import List, Callable, Union

def compute_target_qvalues(
        batch: ExperienceBatch, 
        critic1: NHeadCritic, 
        critic2: NHeadCritic, 
        actor: NHeadActor,
        reward_decay=0.99,
        entropy_weight=0.2
    ):
    with torch.no_grad():
        c1_next_state_qvalues = critic1(batch.next_states)
        c2_next_state_qvalues = critic2(batch.next_states)
        next_action_dist = actor(batch.next_states)
        
    next_state_q = torch.minimum(c1_next_state_qvalues, c2_next_state_qvalues)
    next_state_q = next_state_q - entropy_weight * next_action_dist.logits
    next_state_values = (next_action_dist.probs * next_state_q).sum(-1)
        
    return batch.rewards + (1.0 - batch.dones) * reward_decay * next_state_values
        

class DiscreteMultivationSAC:
    def __init__(self, 
            actor: NHeadActor, 
            critic_template: Callable[[], NHeadCritic], 
            reward_sources: List[RewardGenerator],
            memory: ReplayMemory,
            reward_decay: float=0.99,
            entropy_weight: float=0.2,
            learning_rate: float=0.003,
            device: Union[torch.device, str]="cpu"
        ):
        self.memory = memory
        self.device = torch.device(device)
        self.entropy_weight = entropy_weight
        self.reward_decay = reward_decay
        
        self.actor = actor.to(device)
        self.local_critic_1 = critic_template().to(device)
        self.local_critic_2 = critic_template().to(device)
        self.target_critic_1 = critic_template().to(device)
        self.target_critic_2 = critic_template().to(device)
        
        self.reward_sources = reward_sources
        assert len(self.reward_sources) == self.actor.num_heads, "The number of actor-heads do not match the number of reward generators."
        assert len(self.reward_sources) == self.local_critic_1.num_heads, "The number of critic-heads do not match the number of reward generators."
        
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(params=self.local_critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(params=self.local_critic_2.parameters(), lr=learning_rate)
        
        # Ensure local and target critics are the same
        copy_parameters(self.local_critic_1, self.target_critic_1)
        copy_parameters(self.local_critic_2, self.target_critic_2)
        
    def train(self,
            env: gym.Env,
            total_steps: int,
            initialisation_steps: int=20000,
            update_interval: int=100,
            batch_size: int=64,
            update_steps: int=4,
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
            
    def learn(self, batch_size: int=64):
        """
        Performs one update step. For this, we first sample from the ReplayMemory and then use the data to perform the following steps:
        1. Perform one gradient step on the local critics to better approximate the Q-function.
        2. Perform one gradient step on the policy to act better in the environment.
        3. Interpolate the target critics parameters towards the local critics using polyak averaging.
        """
        # Sample training data
        batch = self.memory.sample(batch_size, device=self.device)
        rewards = [source.generate_rewards(batch) for source in self.reward_sources]
        rewards = torch.stack(rewards, dim=0)
        batch = batch._replace(rewards=rewards)
        
        # Compute critic losses
        target_qvalues = compute_target_qvalues(
            batch=batch, 
            critic1=self.target_critic_1, 
            critic2=self.target_critic_2, 
            actor=self.actor,
            reward_decay=self.reward_decay,
            entropy_weight=self.entropy_weight
        )
        
        critic1_state_qvalues = self.local_critic_1(batch.states)
        critic2_state_qvalues = self.local_critic_2(batch.states)
        
        critic1_loss = F.mse_loss(critic1_state_qvalues, target_qvalues)
        critic2_loss = F.mse_loss(critic2_state_qvalues, target_qvalues)
        
        # Compute actor loss
        critic1_state_qvalues = critic1_state_qvalues.detach()
        critic2_state_qvalues = critic2_state_qvalues.detach()
        state_qvalues = torch.minimum(critic1_state_qvalues, critic2_state_qvalues)
        
        action_dist = self.actor(batch.states)
        values = (action_dist.probs * state_qvalues).sum(-1)
    
    def sample_actions(self, states: torch.FloatTensor, head_weightings: List[float]) -> torch.LongTensor:
        with torch.no_grad():
            dist = self.actor(states)
            head_probs = dist.probs
            
        # Combine probabilities using the weighting
        for i, weighting in enumerate(head_weightings):
            head_probs[i] *= weighting
        combined_probs = head_probs.sum(axis=0)
        
        return torch.distributions.Categorical(probs=combined_probs).sample()