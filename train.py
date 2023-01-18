import argparse
import numpy as np
import gym


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

eval_count = 0
def evaluate_multivation_agent(eval_env: gym.Env, agent: DiscreteMultivationSAC, logger: SummaryWriter, n_episodes=10):
    for head_index in range(agent.num_heads):
        state = eval_env.reset()
        done = False
        
        rewards = []
        predictions = []
        screens = []
        while not done:
            action = agent.predict_head([state], head_index)[0]
            prediction = agent.local_critic_1.predict_head(torch.FloatTensor([state]), head_index)[0][action].item()
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            
            rewards.append(reward)
            predictions.append(prediction)
            screens.append(eval_env.render("rgb_array").transpose(2, 0, 1))
        
        # Log results
        global eval_count
        eval_count+=1
        logger.add_scalar(f"eval/mean_reward_{head_index}", np.sum(rewards), global_step=eval_count)
        logger.add_video(f"eval/video_{head_index}", torch.ByteTensor(screens).unsqueeze(0), fps=40, global_step=eval_count)
        
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
    #logger = None
    
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
    initialisation_steps = 20000
    while agent.total_steps < args.train_steps:
        agent.train(
            env,
            total_steps=args.evaluation_interval,
            initialisation_steps=initialisation_steps,
            logger=logger
        )
        initialisation_steps = 0
        
        evaluate_multivation_agent(
            eval_env=eval_env,
            agent=agent,
            logger=logger
        )