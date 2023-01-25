import numpy as np

import gym
import gym.wrappers as wrappers

def make_preprocessed_env(env_id, normalize_reward=True):
    env = gym.make(env_id, render_mode="rgb_array")
    if is_atari(env_id):
        env = wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=64, scale_obs=True) # frame_skip already handled by the AtariEnv itself
        env = wrappers.FrameStack(env, 2)
    
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    return env

def is_atari(env_id):
    spec = gym.envs.registry.get(env_id)
    return spec.entry_point == "ale_py.env.gym:AtariEnv"
