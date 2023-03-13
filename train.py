# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import time
import yaml
from distutils.util import strtobool

from rewards import ExtrinsicRewardGenerator, CuriosityRewardGenerator, RNDRewardGenerator
from models import NHeadActor, NHeadCritic
from nhead_sac import NHeadSAC

import gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None,
        help="Path to a optional YAML config file.")
    parser.add_argument("--exp-name", type=str, default="experiment", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, 
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model-frequency", type=int, default=int(1e5),
        help="After how many steps should a copy of the agent be made (see 'models' folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate used for training the agent")
    parser.add_argument("--update-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
        help="coefficient for scaling the autotune entropy target")
    args = parser.parse_args()
    # fmt: on
    return args

def parse_reward_sources(yaml_node, device, env: gym.vector.SyncVectorEnv):
    source_types = {
        "extrinsic": ExtrinsicRewardGenerator,
        "curiosity": CuriosityRewardGenerator,
        "rnd": RNDRewardGenerator
    }
    
    reward_sources = []
    for source_node in yaml_node:
        reward_type = source_node["type"].lower()
        new_source = source_types[reward_type].from_config(source_node, device, env.single_action_space, env.single_observation_space)
        reward_sources.append(new_source)
    return reward_sources

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, shape=env.observation_space.shape, dtype=np.float32)
        
    def observation(self, observation):
        return observation.astype(np.float32) / 255.

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using {device}")
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text("device", str(device))
    model_folder = f"{writer.get_logdir()}/models"
    os.makedirs(model_folder)
    
    # Parse config
    config = None
    if args.config_path:
        with open(args.config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # rewards setup
    if config is None:
        reward_sources = [
            #ExtrinsicRewardGenerator(device),
            #CuriosityRewardGenerator(
            #    device,
            #    state_shape=envs.single_observation_space.shape,
            #    embedding_size=128,
            #    num_actions=envs.single_action_space.n
            #),
            RNDRewardGenerator(
                device,
                state_shape=envs.single_observation_space.shape,
                embedding_size=128,
                num_actions=envs.single_action_space.n,
                use_obs_norm=True,
                use_reward_norm=True
            )
        ]
    else:
        reward_sources = parse_reward_sources(config["RewardSources"], device, envs)
    num_heads = len(reward_sources)
    print(f"Using {num_heads} heads")

    # agent setup
    actor = NHeadActor(num_heads, envs.single_observation_space.shape, envs.single_action_space.n)
    critic_template = lambda: NHeadCritic(num_heads, envs.single_observation_space.shape, envs.single_action_space.n)
        
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    
    agent = NHeadSAC(
        actor=actor, 
        critic_template=critic_template, 
        reward_sources=reward_sources,
        memory=rb,
        num_actions=envs.single_action_space.n,
        entropy_weight=args.alpha,
        autotune_entropy=args.autotune,
        target_entropy_scale=args.target_entropy_scale,
        polyak_weight=args.tau,
        learning_rate=args.learning_rate,
        device=device
    )
    agent.train(
        envs=envs,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        update_frequency=args.update_frequency,
        target_network_frequency=args.target_network_frequency,
        save_model_frequency=args.save_model_frequency,
        writer=writer,
        model_folder=model_folder
    )
                
    envs.close()
    writer.close()
