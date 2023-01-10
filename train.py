import argparse
import gym

from torch.utils.tensorboard import SummaryWriter

from models import NHeadActor, NHeadCritic
from rewards import ExtrinsicRewardGenerator, NegativeOneRewardGenerator
from discrete_multivation_sac import DiscreteMultivationSAC
from memory import ReplayMemory

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
    logger = SummaryWriter(log_dir=f"{args.log_folder}/{args.experiment_name}")
    
    # Initialize environment
    env = gym.make(args.env)
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
    while agent.total_steps < args.train_steps:
        agent.train(
            env,
            total_steps=args.evaluation_interval,
            initialisation_steps=20000
            logger=logger
        )