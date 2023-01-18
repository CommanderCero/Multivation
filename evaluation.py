import gym
import torch
import numpy as np

from discrete_multivation_sac import DiscreteMultivationSAC
from torch.utils.tensorboard import SummaryWriter

class MultivationAgentEvaluator:
    def __init__(self, agent: DiscreteMultivationSAC, eval_env: gym.Env, logger: SummaryWriter, num_episodes=10):
        self.agent = agent
        self.eval_env = eval_env
        self.logger = logger
        self.num_episodes = num_episodes
        self.num_evaluations = 0
        
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
            state = self.eval_env.reset()
            done = False
            
            while not done:
                action = self.agent.predict_head([state], head_index)[0]
                next_state, reward, done, info = self.eval_env.step(action)
                state = next_state
                
                reward_sum += reward
                episode_length += 1
                screens.append(self.eval_env.render("rgb_array").transpose(2, 0, 1))
            
            episode_rewards.append(reward_sum)
            episode_lengths.append(episode_length)
                
        self.logger.add_scalar(f"eval/episode_mean_reward_{head_index}", np.mean(episode_rewards), global_step=self.num_evaluations)
        self.logger.add_scalar(f"eval/episode_mean_length_{head_index}", np.mean(episode_lengths), global_step=self.num_evaluations)
        self.logger.add_video(f"eval/video_{head_index}", torch.ByteTensor(screens).unsqueeze(0), fps=40, global_step=self.num_evaluations)