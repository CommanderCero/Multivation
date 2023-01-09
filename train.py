import gym

from models import NHeadActor, NHeadCritic
from rewards import ExtrinsicRewardGenerator
from discrete_multivation_sac import DiscreteMultivationSAC
from memory import ReplayMemory

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trains a Multivation-SAC model")
    parser.add_argument("-env", default="CartPole-v1", help="The name of the gym environment that the agent should learn to play.")
    parser.add_argument("-train_steps", default=1000000, type=int, help="The total amount of steps the agent can take in the environment.")
    parser.add_argument("-memory_size", default=1000000, type=int, help="The size of the replay memory that the agent uses.")
    args = parser.parse_args()
    
    # Initialize environment
    env = gym.make(args.env)
    assert isinstance(env.action_space, gym.spaces.Discrete), "This implementation of MultivationSAC only supports environments with discrete action spaces"
    
    # Initialize Rewards
    reward_sources = [
        ExtrinsicRewardGenerator(),
        ExtrinsicRewardGenerator()
    ]
    
    # Initialize Agent
    num_actions = env.action_space.n
    actor = NHeadActor.create_pure_feedforward([*env.observation_space.shape, 64, 64], len(reward_sources), [64, num_actions])
    critic_template = lambda: NHeadCritic.create_pure_feedforward([*env.observation_space.shape, 64, 64], len(reward_sources), [64, num_actions])
    memory = ReplayMemory(args.memory_size, env.observation_space, env.action_space)
    agent = DiscreteMultivationSAC(actor, critic_template, reward_sources, memory)
    
    # Train
    agent.train(
        env,
        total_steps=args.train_steps
    )