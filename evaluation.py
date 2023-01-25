import gym
import torch
import numpy as np

from discrete_multivation_sac import DiscreteMultivationSAC
from torch.utils.tensorboard import SummaryWriter

class MultivationAgentEvaluator:
    def __init__(self,
        agent: DiscreteMultivationSAC,
        eval_env: gym.vector.VectorEnv,
        logger: SummaryWriter,
        best_model_folder: str,
        num_episodes=4
    ):
        self.agent = agent
        self.eval_env = eval_env
        self.logger = logger
        self.best_model_folder = best_model_folder
        self.num_episodes = num_episodes
        self.num_evaluations = 0
        
        self.best_rewards = np.array([float('-inf')] * self.agent.num_heads)
        
    def evaluate(self):
        for i in range(self.agent.num_heads):
            self.evaluate_head(i)
        self.num_evaluations += 1
        
    def evaluate_head(self, head_index:int):
        episode_rewards = []
        episode_lengths = []
        
        current_reward_sums = np.zeros((self.eval_env.num_envs,), dtype="float32")
        current_episode_lengths = np.zeros((self.eval_env.num_envs,), dtype=int)
        
        states, _ = self.eval_env.reset()
        episode_count = 0
        while episode_count < self.num_episodes:
            actions = self.agent.predict_head(states, head_index)
            next_states, rewards, dones, truncated, _ = self.eval_env.step(actions)
            states = next_states
            
            current_reward_sums += rewards
            current_episode_lengths += 1
            
            # Handle end of episode
            end_mask = dones | truncated
            episode_rewards.extend(current_reward_sums[end_mask])
            episode_lengths.extend(current_episode_lengths[end_mask])
            current_reward_sums[end_mask] = 0
            current_episode_lengths[end_mask] = 0
            episode_count += end_mask.sum()
        
        self.logger.add_scalar(f"eval/episode_mean_reward_{head_index}", np.mean(episode_rewards), global_step=self.num_evaluations)
        self.logger.add_scalar(f"eval/episode_mean_length_{head_index}", np.mean(episode_lengths), global_step=self.num_evaluations)
        
        if np.mean(episode_rewards) > self.best_rewards[head_index]:
            print(f"Found a new best model for head {head_index}")
            self.best_rewards[head_index] = np.mean(episode_rewards)
            self.agent.save(f"{self.best_model_folder}/best_model_{head_index}.torch")