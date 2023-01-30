import numpy as np
import gym
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
        next_state_pi, next_state_log_pi = actor.get_action_probabilities(batch.next_states)
        
    next_state_q = torch.minimum(c1_next_state_qvalues, c2_next_state_qvalues)
    next_state_q = next_state_q - entropy_weight * next_state_log_pi
    next_state_values = (next_state_pi * next_state_q).sum(-1)
    
    return batch.rewards + reward_decay * (1.0 - batch.dones) * next_state_values
        

class DiscreteMultivationSAC:
    def __init__(self,
            actor: NHeadActor, 
            critic_template: Callable[[], NHeadCritic], 
            reward_sources: List[RewardGenerator],
            memory: ReplayMemory,
            num_actions: int,
            reward_decay: float=0.99,
            entropy_weight: float=0.2,
            autotune_entropy: bool=True,
            target_entropy_scale: float=0.89,
            polyak_weight: float=0.995,
            learning_rate: float=0.0003,
            device: Union[torch.device, str]="cpu"
        ):
        self.memory = memory
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.polyak_weight = polyak_weight
        self.reward_decay = reward_decay
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
        
        self.entropy_weight = entropy_weight
        self.autotune_entropy = autotune_entropy
        if self.autotune_entropy:
            self.target_entropy = -target_entropy_scale * np.log(1 / num_actions)
            self.log_entropy_weight = torch.zeros(1, requires_grad=True, device=self.device)
            self.entropy_weight = self.log_entropy_weight.exp().item()
            self.entropy_optimizer = optim.Adam(params=[self.log_entropy_weight], lr=learning_rate, eps=1e-4)
        
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=learning_rate, eps=1e-4)
        self.critic_1_optimizer = optim.Adam(params=self.local_critic_1.parameters(), lr=learning_rate, eps=1e-4)
        self.critic_2_optimizer = optim.Adam(params=self.local_critic_2.parameters(), lr=learning_rate, eps=1e-4)
        
        # Ensure local and target critics are the same
        copy_parameters(self.local_critic_1, self.target_critic_1)
        copy_parameters(self.local_critic_2, self.target_critic_2)
        
    def train(self,
            env: gym.vector.VectorEnv,
            total_steps: int,
            initialisation_steps: int=20000,
            update_interval: int=4,
            batch_size: int=64,
            update_steps: int=4,
            logger: SummaryWriter=None,
            logging_interval: int=4
        ):
        assert isinstance(env.single_action_space, gym.spaces.Discrete), "DiscreteMultivationSAC only supports environments with a discrete action space."
        
        episode_rewards = []
        episode_lengths = []
        current_reward_sums = np.zeros((env.num_envs,), dtype="float32")
        current_episode_lengths = np.zeros((env.num_envs,), dtype=int)
        
        steps_taken = 0
        last_update_step = 0
        head_weightings = sample_weighting_vector((env.num_envs, self.num_heads), device=self.device)
        states, _ = env.reset()
        while steps_taken < total_steps:
            # Sample a action
            if steps_taken <= initialisation_steps:
                actions = env.action_space.sample()
            else:
                actions = self.sample_actions(torch.from_numpy(states).to(self.device), head_weightings)
                actions = actions.cpu().numpy()
            
            # Take a step in the environment
            next_states, rewards, dones, truncated, _ = env.step(actions)
            self.memory.add(states, actions, next_states, rewards, dones | truncated)
            states = next_states
            
            # Update metrics
            self.total_steps += env.num_envs
            steps_taken += env.num_envs
            current_episode_lengths += 1
            current_reward_sums += rewards
            
            # End of episode, reset episode specific variables
            end_mask = dones | truncated
            episode_rewards.extend(current_reward_sums[end_mask])
            episode_lengths.extend(current_episode_lengths[end_mask])
            current_reward_sums[end_mask] = 0
            current_episode_lengths[end_mask] = 0
            head_weightings[end_mask] = sample_weighting_vector(head_weightings[end_mask].shape, device=self.device)
            
            # Learn
            if steps_taken > initialisation_steps and (steps_taken - last_update_step) >= update_interval:
                for i in range(update_steps):
                    self.learn(batch_size=batch_size, logger=logger)
                    
                last_update_step = steps_taken
              
            # Log episode metrics
            if len(episode_rewards) >= logging_interval and logger is not None:
                print(f"{self.total_steps}: mean_length={np.mean(episode_lengths)} mean_reward={np.mean(episode_rewards)}")
                logger.add_scalar("rollout/episode_mean_reward", np.mean(episode_rewards), global_step=self.total_steps)
                logger.add_scalar("rollout/episode_mean_length", np.mean(episode_lengths), global_step=self.total_steps)
                episode_rewards.clear()
                episode_lengths.clear()
            
    def learn(self, batch_size: int=64, logger: SummaryWriter=None):
        """
        Performs one update step. For this, we first sample from the ReplayMemory and then use the data to perform the following steps:
        1. Perform one gradient step on the local critics to better approximate the Q-function.
        2. Interpolate the target critics parameters towards the local critics using polyak averaging.
        3. Perform one gradient step on the policy to act better in the environment.
        
        Returns a tuple containing (actor_loss, entropy, critic1_loss, critic2_loss)
        """
        self.total_updates += 1
        
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
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        polyak_averaging(self.local_critic_1, self.target_critic_1, tau=self.polyak_weight)
        polyak_averaging(self.local_critic_2, self.target_critic_2, tau=self.polyak_weight)
        
        # Compute actor loss
        state_pi, state_log_pi = self.actor.get_action_probabilities(batch.states)
        with torch.no_grad():
            critic1_state_qvalues = self.local_critic_1(batch.states)
            critic2_state_qvalues = self.local_critic_2(batch.states)
        
        state_values = torch.minimum(critic1_state_qvalues, critic2_state_qvalues)
        actor_loss = torch.mean(state_pi * (self.entropy_weight * state_log_pi - state_values))
        entropies = torch.sum(-state_log_pi * state_pi, dim=-1)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy weight
        if self.autotune_entropy:
            entropy_loss = torch.mean(state_pi.detach() * -self.log_entropy_weight * (state_log_pi.detach() + self.target_entropy))
            
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()
            self.entropy_weight = self.log_entropy_weight.exp().item()
            
        if logger and self.total_updates % 100 == 0:
            logger.add_scalar("learn/num_updates", self.total_updates, global_step=self.total_steps)
            logger.add_scalar("learn/actor_loss", actor_loss.item(), global_step=self.total_steps)
            logger.add_scalar("learn/actor_entropy", entropies.mean().item(), global_step=self.total_steps)
            logger.add_scalar("learn/critic1_loss", critic1_loss.item(), global_step=self.total_steps)
            logger.add_scalar("learn/critic2_loss", critic2_loss.item(), global_step=self.total_steps)
            logger.add_scalar("learn/q1_values", critic1_state_qvalues.mean().item(), global_step=self.total_steps)
            logger.add_scalar("learn/q2_values", critic2_state_qvalues.mean().item(), global_step=self.total_steps)
            logger.add_scalar("learn/entropy_weight", self.entropy_weight, global_step=self.total_steps)
            if self.autotune_entropy:
                logger.add_scalar("learn/entropy_loss", entropy_loss.item(), global_step=self.total_steps)
    
    @torch.inference_mode()
    def predict_head(self, states: np.ndarray, head_index: int, deterministic: bool=False):
        states = torch.from_numpy(states).to(self.device)
        logits = self.actor.forward_head(states, head_index=head_index)
        
        if deterministic:
            actions = logits.argmax(-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        
        return actions.cpu().numpy()
    
    @torch.inference_mode()
    def sample_actions(self, states: torch.FloatTensor, head_weightings: torch.FloatTensor) -> torch.LongTensor:
        """
        states: (num_envs, state_shape)
        head_weightings: (num_envs, num_heads) s.t sum(head_weightings, axis=1) == [1., 1., ...]
        """
        
        "logits: (num_heads, num_envs, num_actions)"
        logits = self.actor(states)
            
        # Combine probabilities using the weighting
        for head_index in range(self.num_heads):
            logits[head_index] *= head_weightings[:, head_index].view(-1, 1)
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
        
    def load(self, file_path, device=None):
        data = torch.load(file_path, map_location=device)
        self.actor.load_state_dict(data["actor"])
        self.local_critic_1.load_state_dict(data["local_critic_1"])
        self.local_critic_2.load_state_dict(data["local_critic_2"])
        self.target_critic_1.load_state_dict(data["target_critic_1"])
        self.target_critic_2.load_state_dict(data["target_critic_2"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(data["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(data["critic_2_optimizer"])