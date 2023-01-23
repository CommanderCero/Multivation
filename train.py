import argparse
import numpy as np
import gym
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

from models import NHeadActor, NHeadCritic
from rewards import ExtrinsicRewardGenerator, NegativeOneRewardGenerator
from discrete_multivation_sac import DiscreteMultivationSAC
from memory import ReplayMemory
from gym_utils import make_preprocessed_env
from evaluation import MultivationAgentEvaluator
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a Multivation-SAC model")
    parser.add_argument("-env", default="CartPole-v1", help="The name of the gym environment that the agent should learn to play.")
    parser.add_argument("-train_steps", default=1000000, type=int, help="The total amount of steps the agent can take in the environment.")
    parser.add_argument("-evaluation_interval", default=10000, type=int, help="The amount of steps to take after which the agent will be evaluated.")
    parser.add_argument("-memory_size", default=100000, type=int, help="The size of the replay memory that the agent uses.")
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
    env = make_preprocessed_env(args.env)
    eval_env = make_preprocessed_env(args.env, normalize_reward=False)
    assert isinstance(env.action_space, gym.spaces.Discrete), "This implementation of MultivationSAC only supports environments with discrete action spaces"
    
    # Initialize Rewards
    reward_sources = [
        ExtrinsicRewardGenerator(),
        #NegativeOneRewardGenerator()
    ]
    
    # Initialize Agent
    num_actions = env.action_space.n
    num_heads = len(reward_sources)
    input_shape = env.observation_space.shape
    head_layers = [64, num_actions]
    if len(env.observation_space.shape) == 1:
        actor = NHeadActor.create_pure_feedforward([*input_shape, 64, 64], num_heads, head_layers)
        critic_template = lambda: NHeadCritic.create_pure_feedforward([*input_shape, 64, 64], num_heads, head_layers)
    else:
        embedding_size = 128
        
        actor = NHeadActor.create_conv(
            input_shape=input_shape,
            embedding_size=embedding_size,
            conv_channels=[32,32,32,32],
            body_layers=[],
            num_heads=num_heads,
            head_layers=head_layers
        )
        critic_template = lambda: NHeadCritic.create_conv(
            input_shape=input_shape,
            embedding_size=embedding_size,
            conv_channels=[32,32,32,32],
            body_layers=[],
            num_heads=num_heads,
            head_layers=head_layers
        )
    
    memory = ReplayMemory(args.memory_size, env.observation_space, env.action_space)
    agent = DiscreteMultivationSAC(actor, critic_template, reward_sources, memory)
    
    # Train
    evaluator = MultivationAgentEvaluator(agent, eval_env, logger, log_folder)
    initialisation_steps = 20000
    while agent.total_steps < args.train_steps:
        agent.train(
            env,
            total_steps=args.evaluation_interval,
            initialisation_steps=initialisation_steps,
            logger=logger
        )
        initialisation_steps = max(initialisation_steps - args.evaluation_interval, 0)
        
        evaluator.evaluate()