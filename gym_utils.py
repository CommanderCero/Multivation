import numpy as np

import gym
import gym.wrappers as wrappers
import gym.vector as vgym               

def make_preprocessed_vec_env(env_id, num_envs, normalize_reward=True):
    fn = lambda: make_preprocessed_env(env_id, normalize_reward=False)
    env = vgym.SyncVectorEnv([fn for _ in range(num_envs)])
    if normalize_reward:
        # Normalization has to be the same across all environments
        env = gym.wrappers.NormalizeReward(env)
    return env

def make_preprocessed_env(env_id, normalize_reward=True, render_mode="rgb_array"):
    env = gym.make(env_id, render_mode=render_mode)
    if is_atari(env_id):
        env = wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=64, scale_obs=True) # frame_skip already handled by the AtariEnv itself
        env = wrappers.FrameStack(env, 2)
    
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    return env

def is_atari(env_id):
    spec = gym.envs.registry.get(env_id)
    return spec.entry_point == "ale_py.env.gym:AtariEnv"
