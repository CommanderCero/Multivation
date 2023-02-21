# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import argparse
import os
import random
import time
import yaml
from distutils.util import strtobool

from rewards import ExtrinsicRewardGenerator, CuriosityRewardGenerator
from models import NHeadActor, NHeadSoftQNetwork

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, 
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model-frequency", type=int, default=int(1e5),
        help="After how many steps should a copy of the agent be made (see 'models' folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e4),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
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
    }
    
    reward_sources = []
    for source_node in yaml_node:
        reward_type = source_node["type"].lower()
        new_source = source_types[reward_type].from_config(source_node, device, env.single_action_space, env.single_observation_space)
        reward_sources.append(new_source)
    return reward_sources

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
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # rewards setup
    if config is None:
        reward_sources = [
            #ExtrinsicRewardGenerator(device),
            CuriosityRewardGenerator(
                device,
                state_shape=envs.single_observation_space.shape,
                embedding_size=128,
                num_actions=envs.single_action_space.n
            )
        ]
    else:
        reward_sources = parse_reward_sources(config["RewardSources"], device, envs)
    num_heads = len(reward_sources)
    print(f"Using {num_heads} heads")

    # agent setup
    actor = NHeadActor(num_heads, envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    qf1 = NHeadSoftQNetwork(num_heads, envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    qf2 = NHeadSoftQNetwork(num_heads, envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    qf1_target = NHeadSoftQNetwork(num_heads, envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    qf2_target = NHeadSoftQNetwork(num_heads, envs.single_observation_space.shape, envs.single_action_space.n).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * np.log(1 / envs.single_action_space.n)
        new_log_alpha = torch.zeros(num_heads, requires_grad=True, device=device)
        new_alpha = new_log_alpha.exp().detach()
        a_optimizer = optim.Adam([new_log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha
        
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    
    # Setup saving models
    def save_agent(file_path):
        data = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            **{f"reward_source_{i}" : reward_source.state_dict() for i, reward_source in enumerate(reward_sources)}
        }
        torch.save(data, file_path)

    def load_agent(file_path, device=None):
        data = torch.load(file_path, map_location=device)
        actor.load_state_dict(data["actor"])
        qf1.load_state_dict(data["qf1"])
        qf2.load_state_dict(data["qf2"])
        qf1_target.load_state_dict(data["qf1_target"])
        qf2_target.load_state_dict(data["qf2_target"])
        actor_optimizer.load_state_dict(data["actor_optimizer"])
        q_optimizer.load_state_dict(data["q_optimizer"])
        for i, reward_source in enumerate(reward_sources):
            reward_source.load_state_dict(data[f"reward_source_{i}"])
    models_folder = f"{writer.get_logdir()}/models"
    os.makedirs(models_folder)
    
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            # envs.action_space.sample() is not deterministic for some reason
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
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
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                
                generator_metrics = {}
                for source in reward_sources:
                    metrics = source.update(data)
                    generator_metrics.update({f"{source.__class__.__name__}/{key}" : value for key, value in metrics.items()})
                
                # Generate data for training each head
                rewards, dones, gammas = zip(*[source.generate_data(data) for source in reward_sources])
                rewards = torch.stack([r.flatten() for r in rewards], dim=0)
                dones = torch.stack([d.flatten() for d in dones], dim=0)
                gammas = torch.FloatTensor(gammas).to(device)
                
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - new_alpha.reshape(-1,1,1) * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=-1)
                    next_q_value = rewards + (1 - dones) * gammas.reshape(-1,1) * min_qf_next_target

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                expanded_actions = data.actions.view(1, -1, 1).expand(num_heads, -1, 1).long()
                qf1_a_values = qf1_values.gather(-1, expanded_actions).squeeze(-1)
                qf2_a_values = qf2_values.gather(-1, expanded_actions).squeeze(-1)
                qf1_losses = torch.mean((qf1_a_values - next_q_value) ** 2, dim=-1)
                qf2_losses = torch.mean((qf2_a_values - next_q_value) ** 2, dim=-1)
                qf_loss = qf1_losses.sum() + qf2_losses.sum()

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_losses = (action_probs * (new_alpha.reshape(-1,1,1) * log_pi - min_qf_values))
                actor_losses = torch.mean(actor_losses.view(num_heads, -1), dim=-1)
                actor_loss = actor_losses.sum()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_losses = (action_probs.detach() * (-new_log_alpha.reshape(-1,1,1) * (log_pi.detach() + target_entropy))).view(num_heads, -1).mean(-1)
                    alpha_loss = alpha_losses.sum()
                    
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    new_alpha = new_log_alpha.exp().detach()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                entropies = -torch.mean((log_pi * action_probs).sum(dim=-1), dim=-1)
                for i in range(num_heads):
                    writer.add_scalar(f"head_{i}/qf1_values", qf1_a_values[i].mean().item(), global_step)
                    writer.add_scalar(f"head_{i}/qf2_values", qf2_a_values[i].mean().item(), global_step)
                    writer.add_scalar(f"head_{i}/qf1_loss", qf1_losses[i].item(), global_step)
                    writer.add_scalar(f"head_{i}/qf2_loss", qf2_losses[i].item(), global_step)
                    writer.add_scalar(f"head_{i}/actor_loss", actor_losses[i].item(), global_step)
                    writer.add_scalar(f"head_{i}/alpha", new_alpha[i], global_step)
                    writer.add_scalar(f"head_{i}/entropy", entropies[i].cpu().item(), global_step)
                    if args.autotune:
                        writer.add_scalar(f"head_{i}/alpha_loss", alpha_losses[i].item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                for name, metric in generator_metrics.items():
                    writer.add_scalar(name, metric, global_step)
                    
                for i, head_rewards in enumerate(rewards):
                    writer.add_scalar(f"head_{i}/mean_rewards", head_rewards.mean().item(), global_step)
                    writer.add_scalar(f"head_{i}/min_rewards", head_rewards.min().item(), global_step)
                    writer.add_scalar(f"head_{i}/max_rewards", head_rewards.max().item(), global_step)
                    
        if global_step % args.save_model_frequency == 0:
            print(f"{global_step}: Saving agent...")
            save_agent(f"{models_folder}/model_{global_step // args.save_model_frequency}.torch")
                
    envs.close()
    writer.close()
