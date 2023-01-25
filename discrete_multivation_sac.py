import numpy as np
import gym
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_utils import copy_parameters, sample_weighting_vector, polyak_averaging
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
        next_action_dist = actor.get_action_dist(batch.next_states)
        
    next_state_q = torch.minimum(c1_next_state_qvalues, c2_next_state_qvalues)
    next_state_q = next_state_q - entropy_weight * next_action_dist.logits
    next_state_values = (next_action_dist.probs * next_state_q).sum(-1)
    
    return batch.rewards + reward_decay * (1.0 - batch.dones) * next_state_values
        

class DiscreteMultivationSAC:
    def __init__(self,
            actor: NHeadActor, 
            critic_template: Callable[[], NHeadCritic], 
            reward_sources: List[RewardGenerator],
            memory: ReplayMemory,
            reward_decay: float=0.99,
            entropy_weight: float=0.2,
            polyak_weight: float=0.995,
            learning_rate: float=0.0003,
            device: Union[torch.device, str]="cpu"
        ):
        self.memory = memory
        self.device = torch.device(device)
        self.polyak_weight = polyak_weight
        self.reward_decay = reward_decay
        self.entropy_weight = entropy_weight
        self.total_steps = 0
        self.total_updates = 0
        
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
            update_interval: int=4,
            batch_size: int=64,
            update_steps: int=4,
            logger: torch.utils.tensorboard.SummaryWriter=None,
            logging_interval: int=4
        ):
        assert isinstance(env.action_space, gym.spaces.Discrete), "DiscreteMultivationSAC only supports environments with a discrete action space."
        
        episode_rewards = []
        episode_lengths = []
        reward_sum = 0
        episode_length = 0
        
        steps_taken = 0
        head_weightings = sample_weighting_vector(self.num_heads)
        state, _ = env.reset()
        while steps_taken < total_steps:
            # Sample a action
            if steps_taken <= initialisation_steps:
                action = env.action_space.sample()
            else:
                action = self.sample_actions(torch.Tensor([state]), head_weightings).item()
            
            # Take a step in the environment
            next_state, reward, done, truncated, info = env.step(action)
            self.memory.add(state, action, next_state, reward, done)
            state = next_state
            
            # Update metrics
            steps_taken += 1
            self.total_steps += 1
            episode_length += 1
            reward_sum += reward
            
            # End of episode, reset episode specific variables
            if done or truncated:
                state, _ = env.reset()
                head_weightings = sample_weighting_vector(self.num_heads)
                
                episode_rewards.append(reward_sum)
                episode_lengths.append(episode_length)
                reward_sum = 0
                episode_length = 0
            
            # Learn
            if steps_taken > initialisation_steps and self.total_steps % update_interval == 0:
                actor_losses = []
                entropies = []
                critic1_losses = []
                critic2_losses = []
                
                for i in range(update_steps):
                    actor_loss, entropy, critic1_loss, critic2_loss = self.learn(batch_size=batch_size)
                    self.total_updates += 1
                    
                    actor_losses.append(actor_loss)
                    entropies.append(entropy)
                    critic1_losses.append(critic1_loss)
                    critic2_losses.append(critic2_loss)
                    
                # Log training results
                if logger:
                    logger.add_scalar("learn/num_updates", self.total_updates, global_step=self.total_steps)
                    logger.add_scalar("learn/actor_loss", np.mean(actor_losses), global_step=self.total_steps)
                    logger.add_scalar("learn/actor_entropy", np.mean(entropies), global_step=self.total_steps)
                    logger.add_scalar("learn/critic1_loss", np.mean(critic1_losses), global_step=self.total_steps)
                    logger.add_scalar("learn/critic2_loss", np.mean(critic2_losses), global_step=self.total_steps)
              
            # Log episode metrics
            if len(episode_rewards) >= logging_interval:
                print(f"{self.total_steps}: mean_length={np.mean(episode_lengths)} mean_reward={np.mean(episode_rewards)}")
                logger.add_scalar("rollout/episode_mean_reward", np.mean(episode_rewards), global_step=self.total_steps)
                logger.add_scalar("rollout/episode_mean_length", np.mean(episode_lengths), global_step=self.total_steps)
                episode_rewards.clear()
                episode_lengths.clear()
            
    def learn(self, batch_size: int=64):
        """
        Performs one update step. For this, we first sample from the ReplayMemory and then use the data to perform the following steps:
        1. Perform one gradient step on the local critics to better approximate the Q-function.
        2. Interpolate the target critics parameters towards the local critics using polyak averaging.
        3. Perform one gradient step on the policy to act better in the environment.
        
        Returns a tuple containing (actor_loss, entropy, critic1_loss, critic2_loss)
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
        
        expanded_actions = batch.actions.view(1, -1, 1).expand(self.num_heads, -1, 1)
        critic1_selected_qvalues = torch.gather(critic1_state_qvalues, dim=-1, index=expanded_actions).squeeze(-1)
        critic2_selected_qvalues = torch.gather(critic2_state_qvalues, dim=-1, index=expanded_actions).squeeze(-1)
        
        critic1_loss = F.mse_loss(critic1_selected_qvalues, target_qvalues)
        critic2_loss = F.mse_loss(critic2_selected_qvalues, target_qvalues)
        
        # Update critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        torch.autograd.backward([critic1_loss, critic2_loss])
        nn.utils.clip_grad_norm_(self.local_critic_1.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.local_critic_2.parameters(), 0.5)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        polyak_averaging(self.local_critic_1, self.target_critic_1, tau=self.polyak_weight)
        polyak_averaging(self.local_critic_2, self.target_critic_2, tau=self.polyak_weight)
        
        # Compute actor loss
        action_dist = self.actor.get_action_dist(batch.states)
        with torch.no_grad():
            critic1_state_qvalues = self.local_critic_1(batch.states)
            critic2_state_qvalues = self.local_critic_2(batch.states)
        
        state_values = torch.minimum(critic1_state_qvalues, critic2_state_qvalues)
        state_values = (action_dist.probs * state_values).sum(-1)
        entropies = action_dist.entropy()
        actor_loss = (-state_values - self.entropy_weight * entropies).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return actor_loss.item(), entropies.mean().item(), critic1_loss.item(), critic2_loss.item()
    
    def predict_head(self, states: np.ndarray, head_index: int, deterministic: bool=False):
        with torch.no_grad():
            logits = self.actor.predict_head(torch.FloatTensor(states), head_index=head_index)
        
        if deterministic:
            actions = logits.argmax(-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        
        return actions.numpy()
    
    def sample_actions(self, states: torch.FloatTensor, head_weightings: List[float]) -> torch.LongTensor:
        with torch.no_grad():
            logits = self.actor(states)
            
        # Combine probabilities using the weighting
        for i, weighting in enumerate(head_weightings):
            logits[i] *= weighting
        combined_logits = logits.sum(axis=0)
        
        return torch.distributions.Categorical(logits=combined_logits).sample()
    
    @property
    def num_heads(self):
        return self.actor.num_heads
    
    def save(self, file_path):
        data = {
            "actor": self.actor.state_dict(),
            "local_critic_1": self.local_critic_1.state_dict(),
            "local_critic_2": self.local_critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
        }
        torch.save(data, file_path)
        
    def load(self, file_path):
        data = torch.load(file_path)
        self.actor.load_state_dict(data["actor"])
        self.local_critic_1.load_state_dict(data["local_critic_1"])
        self.local_critic_2.load_state_dict(data["local_critic_2"])
        self.target_critic_1.load_state_dict(data["target_critic_1"])
        self.target_critic_2.load_state_dict(data["target_critic_2"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(data["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(data["critic_2_optimizer"])