# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import yaml

from stable_baselines3.common.buffers import ReplayBufferSamples
from rewards import ExtrinsicRewardGenerator, CuriosityRewardGenerator
from models import NHeadActor, NHeadSoftQNetwork

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None,
        help="Path to a optional YAML config file.")
    parser.add_argument("--seed", type=int, default=1, 
        help="seed of the experiment")
    args = parser.parse_args()
    # fmt: on
    return args

def parse_reward_sources(yaml_node, device, env):
    source_types = {
        "extrinsic": ExtrinsicRewardGenerator,
        "curiosity": CuriosityRewardGenerator,
    }
    
    reward_sources = []
    for source_node in yaml_node:
        reward_type = source_node["type"].lower()
        new_source = source_types[reward_type].from_config(source_node, device, env.action_space, env.observation_space)
        reward_sources.append(new_source)
    return reward_sources

def make_env(env_id, seed):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

if __name__ == "__main__":
    args = parse_args()
    
    # TRY NOT TO MODIFY: seeding
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True

    device = torch.device("cpu")

    # env setup
    def parse_folder_name(folder_path):
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split("__")
        return (parts[0], parts[1].split("_")[1])
    
    model_folder = "./runs/IceHockeyNoFrameskip-v4__icehockey_extrinsic__2__1677133167"
    game_name, config_type = parse_folder_name(model_folder)
    env = make_env(game_name, args.seed)

    with open(f"configs/{config_type}.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    reward_sources = parse_reward_sources(config["RewardSources"], device, env)
    num_heads = len(reward_sources)
    print(f"Using {num_heads} heads")

    # agent setup
    actor = NHeadActor(num_heads, env.observation_space.shape, env.action_space.n).to(device)
    qf1 = NHeadSoftQNetwork(num_heads, env.observation_space.shape, env.action_space.n).to(device)
    qf2 = NHeadSoftQNetwork(num_heads, env.observation_space.shape, env.action_space.n).to(device)
    qf1_target = NHeadSoftQNetwork(num_heads, env.observation_space.shape, env.action_space.n).to(device)
    qf2_target = NHeadSoftQNetwork(num_heads, env.observation_space.shape, env.action_space.n).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    def load_agent(file_path, device=None):
        data = torch.load(file_path, map_location=device)
        actor.load_state_dict(data["actor"])
        qf1.load_state_dict(data["qf1"])
        qf2.load_state_dict(data["qf2"])
        qf1_target.load_state_dict(data["qf1_target"])
        qf2_target.load_state_dict(data["qf2_target"])
        for i, reward_source in enumerate(reward_sources):
            reward_source.load_state_dict(data[f"reward_source_{i}"])
    
    
    model_file = f"{model_folder}/models/model_49.torch"
    load_agent(model_file, device=device)
    state = env.reset()
    
    frames = [env.render("rgb_array")]
    states = []
    actions = []
    next_states = []
    dones = []
    rewards = []
    
    # Play
    done = False
    while not done:
        a, log_probs, action_probs = actor.get_action(torch.from_numpy(np.array([state])))
        next_state, reward, done, _ = env.step(a.item())
        #next_state, reward, done, _ = env.step(torch.argmax(action_probs).item())
        
        states.append(state)
        next_states.append(next_state)
        actions.append(a.item())
        dones.append(done)
        rewards.append(reward)
        
        frames.append(env.render(mode="rgb_array"))
        state = next_state
    
    # Compute rewards
    data = ReplayBufferSamples(
        observations=torch.from_numpy(np.array(states, dtype=np.float32)),
        next_observations=torch.from_numpy(np.array(next_states, dtype=np.float32)),
        actions=torch.from_numpy(np.array(actions, dtype=np.float32)),
        rewards=torch.from_numpy(np.array(rewards, dtype=np.float32)),
        dones=torch.from_numpy(np.array(dones, dtype=np.float32))
    )
    rewards = reward_sources[0].generate_rewards(data).numpy()
    plt.plot(rewards)
    
    # Create video
    import cv2
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    fps = 30
    output_file = "agent.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()