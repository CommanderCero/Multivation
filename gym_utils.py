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
    env = gym.wrappers.StepAPICompatibility(env, output_truncation_bool=False)
    env = ResetAPICompatibility(env)
    return env

def is_atari(env_id):
    spec = gym.envs.registry.get(env_id)
    return spec.entry_point == "ale_py.env.gym:AtariEnv"
    
class ResetAPICompatibility(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self):
        state, info = self.env.reset()
        return state
    
"""    
env = make_preprocessed_env("CartPole-v1")
for i in range(100):
    done = False
    state = env.reset()
    rewards = []
    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())
        rewards.append(reward)
        
    print(np.sum(rewards), np.mean(rewards), np.std(rewards))
"""