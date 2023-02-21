from models import OneHotForwardModelResiduals, DiscreteActionPredictor, Conv2DEmbedding
from sklearn.metrics import accuracy_score
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import torch
import gym
import numpy as np

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

num_steps = 64
env = make_env("PongNoFrameskip-v4", 0)()
def sample_data(num_steps):
    states = np.empty((num_steps, *env.observation_space.shape), dtype=np.float32)
    next_states = np.empty((num_steps, *env.observation_space.shape), dtype=np.float32)
    actions = np.empty((num_steps, 1), dtype=int)
    done = True
    
    for i in range(num_steps):
        if done:
            obs = env.reset()
            
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        states[i] = obs
        next_states[i] = next_obs
        actions[i] = action
        
        obs = next_obs
    return states, next_states, actions

if False:
    all_states, all_next_states, all_actions = sample_data(1000)
    import matplotlib.pyplot as plt
    while True:
        idx = np.random.randint(0, all_states.shape[0])
        state = all_states[idx]
        next_state = all_next_states[idx]
        
        f, axarr = plt.subplots(2, 4)
        for ax in [ax for arr in axarr for ax in arr]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        axarr[0,0].imshow(state[0])
        axarr[0,1].imshow(state[1])
        axarr[0,2].imshow(state[2])
        axarr[0,3].imshow(state[3])
        
        axarr[1,0].imshow(next_state[0])
        axarr[1,1].imshow(next_state[1])
        axarr[1,2].imshow(next_state[2])
        axarr[1,3].imshow(next_state[3])
        plt.show()
        
        a = int(input())
        print(f"Correct answer was {all_actions[idx]}")

# Train on data
embedding_size = 128
embedding_net = Conv2DEmbedding(env.observation_space.shape, embedding_size)
fm = OneHotForwardModelResiduals(embedding_size, 128, env.action_space.n)
inv_fm = DiscreteActionPredictor(embedding_size, env.action_space.n)

optim = torch.optim.Adam([*embedding_net.parameters(), *fm.parameters(), *inv_fm.parameters()], lr=0.0003)
epochs = 1000
all_states, all_next_states, all_actions = sample_data(1000)
for epoch in range(epochs):
    indices = np.random.randint(0, all_states.shape[0], size=64)
    states = all_states[indices]
    next_states = all_next_states[indices]
    actions = all_actions[indices]
    
    embedded_states = embedding_net(torch.from_numpy(states))
    embedded_next_states = embedding_net(torch.from_numpy(next_states))
    pred_actions = inv_fm(embedded_states, embedded_next_states).argmax(dim=-1)
    
    fm_loss = fm.compute_loss(embedded_states.detach(), torch.from_numpy(actions), embedded_next_states.detach())
    inv_fm_loss = inv_fm.compute_loss(embedded_states, embedded_next_states, torch.from_numpy(actions))
    loss = inv_fm_loss + fm_loss
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    entropy = torch.distributions.Categorical(logits=inv_fm(embedded_states, embedded_next_states)).entropy().mean()
    accuracy = accuracy_score(actions.flatten(), pred_actions.numpy())
    print(f"{epoch} FM: {fm_loss.item():.4f}\tInverse: {inv_fm_loss.item():.4f}\tInv Accuracy: {entropy}")

if False:
    # Train on data
    embedding_size = 128
    embedding_net = Conv2DEmbedding(env.observation_space.shape, embedding_size)
    fm = OneHotForwardModelResiduals(embedding_size, 128, env.action_space.n)
    inv_fm = DiscreteActionPredictor(embedding_size, env.action_space.n)
    
    optim = torch.optim.Adam([*embedding_net.parameters(), *fm.parameters(), *inv_fm.parameters()], lr=0.0003)
    epochs = 1000
    for epoch in range(epochs):
        states, next_states, actions = sample_data(num_steps)
        
        embedded_states = embedding_net(torch.from_numpy(states))
        embedded_next_states = embedding_net(torch.from_numpy(next_states))
        
        fm_loss = fm.compute_loss(embedded_states.detach(), torch.from_numpy(actions), embedded_next_states.detach())
        inv_fm_loss = inv_fm.compute_loss(embedded_states, embedded_next_states, torch.from_numpy(actions))
        loss = inv_fm_loss
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        print(f"{epoch} FM: {fm_loss.item():.4f}\tInverse: {inv_fm_loss.item():.4f}")
    
if False:
    states = np.random.uniform(size=(num_steps, 4, 84, 84)).astype(np.float32)
    actions = np.random.randint(0, env.action_space.n, size=(num_steps, 1))
    next_states = states.copy().astype(np.float32)
    for i, a in enumerate(actions):
        next_states[i] *= a
    
    embedding_size = 128
    embedding_net = Conv2DEmbedding(states.shape[1:], embedding_size)
    inv_fm = DiscreteActionPredictor(embedding_size, env.action_space.n)
    optim = torch.optim.Adam([*embedding_net.parameters(), *inv_fm.parameters()], lr=0.0003)
    train_steps = 1000
    for i in range(train_steps):
        embedded_states = embedding_net(torch.from_numpy(states))
        embedded_next_states = embedding_net(torch.from_numpy(next_states))
        inv_fm_loss = inv_fm.compute_loss(embedded_states, embedded_next_states, torch.from_numpy(actions))
        
        optim.zero_grad()
        inv_fm_loss.backward()
        optim.step()
        
        print(i, inv_fm_loss.item())

if False:
    size = 10000
    all_states = np.random.uniform(size=(size, 4, 84, 84)).astype(np.float32)
    all_actions = np.random.randint(0, env.action_space.n, size=(size, 1))
    all_next_states = np.random.uniform(size=(size, 4, 84, 84), low=0, high=0.1).astype(np.float32) + all_states
    for i, a in enumerate(all_actions):
        all_next_states[i] *= a
    
    embedding_size = 128
    embedding_net = Conv2DEmbedding(all_states.shape[1:], embedding_size)
    inv_fm = DiscreteActionPredictor(embedding_size, env.action_space.n)
    optim = torch.optim.Adam([*embedding_net.parameters(), *inv_fm.parameters()], lr=0.0003)
    train_steps = 1000
    for i in range(train_steps):
        indices = np.random.randint(0, all_states.shape[0], size=64)
        states = all_states[indices]
        next_states = all_next_states[indices]
        actions = all_actions[indices]
        
        embedded_states = embedding_net(torch.from_numpy(states))
        embedded_next_states = embedding_net(torch.from_numpy(next_states))
        pred_actions = inv_fm(embedded_states, embedded_next_states).argmax(dim=-1)
        inv_fm_loss = inv_fm.compute_loss(embedded_states, embedded_next_states, torch.from_numpy(actions))
        
        optim.zero_grad()
        inv_fm_loss.backward()
        optim.step()
        
        accuracy = accuracy_score(actions.flatten(), pred_actions.numpy())
        print(i, inv_fm_loss.item(), accuracy)
