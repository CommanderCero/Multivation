from models import NHeadActor, NHeadCritic
from rewards import RewardGenerator

import torch
import torch.optim as optim

import gym
import time
import numpy as np
import os
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, List, Union, Optional

class NHeadSAC:
    def __init__(self,
            actor: NHeadActor, 
            critic_template: Callable[[], NHeadCritic], 
            reward_sources: List[RewardGenerator],
            memory: ReplayBuffer,
            num_actions: int,
            entropy_weight: float=0.2,
            autotune_entropy: bool=True,
            target_entropy_scale: float=0.89,
            polyak_weight: float=0.995,
            learning_rate: float=0.0003,
            device: Union[torch.device, str]="cpu"
        ):
        self.memory = memory
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.polyak_weight = polyak_weight
        
        self.actor = actor.to(self.device)
        self.critic_1 = critic_template().to(self.device)
        self.critic_2 = critic_template().to(self.device)
        self.target_critic_1 = critic_template().to(self.device)
        self.target_critic_2 = critic_template().to(self.device)
        # Ensure local and target critics are the same
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=learning_rate, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate, eps=1e-4)
        
        self.reward_sources = reward_sources
        assert len(self.reward_sources) == self.actor.num_heads, "The number of actor-heads do not match the number of reward generators."
        assert len(self.reward_sources) == self.critic_1.num_heads, "The number of critic-heads do not match the number of reward generators."
        self.num_heads = len(self.reward_sources)
        
        self.entropy_weight = entropy_weight
        self.autotune_entropy = autotune_entropy
        if autotune_entropy:
            self.target_entropy = -target_entropy_scale * np.log(1 / num_actions)
            self.log_alpha = torch.zeros(self.num_heads, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate, eps=1e-4)
        else:
            self.alpha = entropy_weight
    
    def train(self,
        envs: gym.vector.VectorEnv,
        total_timesteps: int,
        learning_starts: int,
        batch_size: int,
        update_frequency: int,
        target_network_frequency: int,
        save_model_frequency: int,
        writer: SummaryWriter
    ):
        start_time = time.time()
        obs = envs.reset()
        for global_step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts:
                # envs.action_space.sample() is not deterministic for some reason
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions[0] # Head 0
                actions = actions.detach().cpu().numpy()
    
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)
    
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
    
            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.memory.add(obs, real_next_obs, actions, rewards, dones, infos)
            
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            
            # ALGO LOGIC: training.
            if global_step > learning_starts:
                if global_step % update_frequency == 0:
                    data = self.memory.sample(batch_size)
                    if global_step % 100 == 0:
                        self.learn(data, writer=writer, logging_step=global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    else:
                        self.learn(data)
    
                # update the target networks
                if global_step % target_network_frequency == 0:
                    self.update_target_networks()
                        
            if global_step % save_model_frequency == 0:
                print(f"{global_step}: Saving agent...")
                os.makedirs(f"{writer.get_logdir()}/models")
                self.save(f"{writer.get_logdir()}/models/model_{global_step // save_model_frequency}.torch")
    
    def learn(self, data: ReplayBufferSamples, writer: SummaryWriter=None, logging_step=None):
        generator_metrics = {}
        for source in self.reward_sources:
            metrics = source.update(data)
            generator_metrics.update({f"{source.__class__.__name__}/{key}" : value for key, value in metrics.items()})
        
        # Generate data for training each head
        rewards, dones, gammas = zip(*[source.generate_data(data) for source in self.reward_sources])
        rewards = torch.stack([r.flatten() for r in rewards], dim=0)
        dones = torch.stack([d.flatten() for d in dones], dim=0)
        gammas = torch.FloatTensor(gammas).to(self.device)
        
        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(data.next_observations)
            qf1_next_target = self.target_critic_1(data.next_observations)
            qf2_next_target = self.target_critic_2(data.next_observations)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha.reshape(-1,1,1) * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=-1)
            next_q_value = rewards + (1 - dones) * gammas.reshape(-1,1) * min_qf_next_target

        # use Q-values only for the taken actions
        qf1_values = self.critic_1(data.observations)
        qf2_values = self.critic_2(data.observations)
        expanded_actions = data.actions.view(1, -1, 1).expand(self.num_heads, -1, 1).long()
        qf1_a_values = qf1_values.gather(-1, expanded_actions).squeeze(-1)
        qf2_a_values = qf2_values.gather(-1, expanded_actions).squeeze(-1)
        qf1_losses = torch.mean((qf1_a_values - next_q_value) ** 2, dim=-1)
        qf2_losses = torch.mean((qf2_a_values - next_q_value) ** 2, dim=-1)
        qf_loss = qf1_losses.sum() + qf2_losses.sum()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(data.observations)
        with torch.no_grad():
            qf1_values = self.critic_1(data.observations)
            qf2_values = self.critic_2(data.observations)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_losses = (action_probs * (self.alpha.reshape(-1,1,1) * log_pi - min_qf_values))
        actor_losses = torch.mean(actor_losses.view(self.num_heads, -1), dim=-1)
        actor_loss = actor_losses.sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune_entropy:
            # re-use action probabilities for temperature loss
            alpha_losses = (action_probs.detach() * (-self.log_alpha.reshape(-1,1,1) * (log_pi.detach() + self.target_entropy))).view(self.num_heads, -1).mean(-1)
            alpha_loss = alpha_losses.sum()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
            
        # Logging
        if writer is not None:
            entropies = -torch.mean((log_pi * action_probs).sum(dim=-1), dim=-1)
            for i in range(self.num_heads):
                writer.add_scalar(f"head_{i}/qf1_values", qf1_a_values[i].mean().item(), logging_step)
                writer.add_scalar(f"head_{i}/qf2_values", qf2_a_values[i].mean().item(), logging_step)
                writer.add_scalar(f"head_{i}/qf1_loss", qf1_losses[i].item(), logging_step)
                writer.add_scalar(f"head_{i}/qf2_loss", qf2_losses[i].item(), logging_step)
                writer.add_scalar(f"head_{i}/actor_loss", actor_losses[i].item(), logging_step)
                writer.add_scalar(f"head_{i}/alpha", self.alpha[i], logging_step)
                writer.add_scalar(f"head_{i}/entropy", entropies[i].cpu().item(), logging_step)
                if self.autotune_entropy:
                    writer.add_scalar(f"head_{i}/alpha_loss", alpha_losses[i].item(), logging_step)
            
            for name, metric in generator_metrics.items():
                writer.add_scalar(name, metric, logging_step)
                
            for i, head_rewards in enumerate(rewards):
                writer.add_scalar(f"head_{i}/mean_rewards", head_rewards.mean().item(), logging_step)
                writer.add_scalar(f"head_{i}/min_rewards", head_rewards.min().item(), logging_step)
                writer.add_scalar(f"head_{i}/max_rewards", head_rewards.max().item(), logging_step)
            
            
    def update_target_networks(self):
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.polyak_weight * param.data + (1 - self.polyak_weight) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.polyak_weight * param.data + (1 - self.polyak_weight) * target_param.data)
    
    def save(self, file_path: str):
        data = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            **{f"reward_source_{i}" : reward_source.state_dict() for i, reward_source in enumerate(self.reward_sources)}
        }
        torch.save(data, file_path)
        
        
    def load(self, file_path: str, device: Optional[torch.device]=None):
        data = torch.load(file_path, map_location=device)
        self.actor.load_state_dict(data["actor"])
        self.critic_1.load_state_dict(data["critic_1"])
        self.critic_2.load_state_dict(data["critic_2"])
        self.target_critic_1.load_state_dict(data["target_critic_1"])
        self.target_critic_2.load_state_dict(data["target_critic_2"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        for i, reward_source in enumerate(self.reward_sources):
            reward_source.load_state_dict(data[f"reward_source_{i}"])