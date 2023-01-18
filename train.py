import argparse
import numpy as np
import gym
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

# DEBUG CODE for getting matplotlib.pyplot.imshow to work
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# I have no idea what exactly causes this error, but I think it's because of pytorch
# In any case I just need matplotlib for debugging
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from models import NHeadActor, NHeadCritic
from rewards import ExtrinsicRewardGenerator, NegativeOneRewardGenerator
from discrete_multivation_sac import DiscreteMultivationSAC
from memory import ReplayMemory

class MultivationAgentEvaluator:
    def __init__(self, agent: DiscreteMultivationSAC, logger: SummaryWriter, num_episodes=10):
        self.agent = agent
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
            state = eval_env.reset()
            done = False
            
            while not done:
                action = agent.predict_head([state], head_index)[0]
                next_state, reward, done, info = eval_env.step(action)
                state = next_state
                
                reward_sum += reward
                episode_length += 1
                screens.append(eval_env.render("rgb_array").transpose(2, 0, 1))
            
            episode_rewards.append(reward_sum)
            episode_lengths.append(episode_length)
                
        logger.add_scalar("eval/episode_mean_reward_{head_index}", np.mean(episode_rewards), global_step=self.num_evaluations)
        logger.add_scalar("eval/episode_mean_length_{head_index}", np.mean(episode_lengths), global_step=self.num_evaluations)
        logger.add_video(f"eval/video_{head_index}", torch.ByteTensor(screens).unsqueeze(0), fps=40, global_step=self.num_evaluations)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a Multivation-SAC model")
    parser.add_argument("-env", default="CartPole-v1", help="The name of the gym environment that the agent should learn to play.")
    parser.add_argument("-train_steps", default=1000000, type=int, help="The total amount of steps the agent can take in the environment.")
    parser.add_argument("-evaluation_interval", default=10000, type=int, help="The amount of steps to take after which the agent will be evaluated.")
    parser.add_argument("-memory_size", default=1000000, type=int, help="The size of the replay memory that the agent uses.")
    parser.add_argument("-log_folder", default="runs", help="Name of the folder where the tensorboard logs are stored.")
    parser.add_argument("-experiment_name", default="experiment", help="Experiment name used for identifying logs in tensorboard.")
    args = parser.parse_args()
    
    # Setup logging
    log_folder = f"{args.log_folder}/{args.experiment_name}"
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    os.makedirs(log_folder)
    
    logger = SummaryWriter(log_dir=log_folder)
    
    # Initialize environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    assert isinstance(env.action_space, gym.spaces.Discrete), "This implementation of MultivationSAC only supports environments with discrete action spaces"
    
    # Initialize Rewards
    reward_sources = [
        ExtrinsicRewardGenerator(),
        #NegativeOneRewardGenerator()
    ]
    
    # Initialize Agent
    num_actions = env.action_space.n
    actor = NHeadActor.create_pure_feedforward([*env.observation_space.shape, 64, 64], len(reward_sources), [64, num_actions])
    critic_template = lambda: NHeadCritic.create_pure_feedforward([*env.observation_space.shape, 64, 64], len(reward_sources), [64, num_actions])
    memory = ReplayMemory(args.memory_size, env.observation_space, env.action_space)
    agent = DiscreteMultivationSAC(actor, critic_template, reward_sources, memory)
    
    # Train
    evaluator = MultivationAgentEvaluator(agent, logger)
    initialisation_steps = 20000
    while agent.total_steps < args.train_steps:
        agent.train(
            env,
            total_steps=args.evaluation_interval,
            initialisation_steps=initialisation_steps,
            logger=logger
        )
        initialisation_steps = 0
        
        evaluator.evaluate()