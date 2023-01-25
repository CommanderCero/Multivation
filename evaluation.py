import gym
import torch
import numpy as np

from discrete_multivation_sac import DiscreteMultivationSAC
from torch.utils.tensorboard import SummaryWriter

class MultivationAgentEvaluator:
    def __init__(self,
        agent: DiscreteMultivationSAC,
        eval_env: gym.Env,
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
        screens = []
        for _ in range(self.num_episodes):
            reward_sum = 0
            episode_length = 0
            state, _ = self.eval_env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                action = self.agent.predict_head(np.array([state]), head_index)[0]
                next_state, reward, done, truncated, info = self.eval_env.step(action)
                state = next_state
                
                reward_sum += reward
                episode_length += 1
                screens.append(self.eval_env.render().transpose(2, 0, 1))
            
            episode_rewards.append(reward_sum)
            episode_lengths.append(episode_length)
        
        self.logger.add_scalar(f"eval/episode_mean_reward_{head_index}", np.mean(episode_rewards), global_step=self.num_evaluations)
        self.logger.add_scalar(f"eval/episode_mean_length_{head_index}", np.mean(episode_lengths), global_step=self.num_evaluations)
        self.logger.add_video(f"eval/video_{head_index}", torch.ByteTensor(screens).unsqueeze(0), fps=40, global_step=self.num_evaluations)
        
        if np.mean(episode_rewards) > self.best_rewards[head_index]:
            print(f"Found a new best model for head {head_index}")
            self.best_rewards[head_index] = np.mean(episode_rewards)
            self.agent.save(f"{self.best_model_folder}/best_model_{head_index}.torch")