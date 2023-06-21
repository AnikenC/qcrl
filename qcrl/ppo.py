import os
import argparse
import time
import random
from tqdm import tqdm
from distutils.util import strtobool

from jax import config
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import wandb

from readout_optimisation.rl_envs.resonator_environment import ResonatorEnv

config.update("jax_enable_x64", True)

# For debugging
# torch.autograd.set_detect_anomaly(True)

### Working! ###
# Run `python3 test_ppo.py --track` to add WandB tracking, make sure WandB has been initialized prior
# Run `python3 test_ppo.py --print-debug` to get print statements in the command line


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="Torched_PPO",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ReadoutOptimisation",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="quantumcontrolwithrl",
        help="the entity (team) of wandb's project")
    parser.add_argument("--print-debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to print debug info in the terminal or not")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="resonator_env",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1, nargs="?", const=True,\
        help="number of environments for parallel processing")
    parser.add_argument("--num-updates", type=int, default=10000, nargs="?", const=True,
        help="total timesteps of the experiments")
    parser.add_argument("--update-epochs", type=int, default=3, nargs="?", const=True,
        help="number of epochs before a policy update")
    parser.add_argument("--num-minibatches", type=int, default=32, nargs="?", const=True,
        help="number of minibatches in one batch")
    parser.add_argument("--learning-rate", type=float, default=0.0003, nargs="?", const=True,
        help="the learning rate of the optimizer")
    parser.add_argument("--batch-size", type=int, default=512, nargs="?", const=True,\
        help="batch size for each update")
    parser.add_argument("--clip-epsilon", type=float, default=0.1, nargs="?", const=True,\
        help="clipping epsilon")
    parser.add_argument("--grad-clip", type=float, default=0.5, nargs="?", const=True,\
        help="gradient clip for optimizer updates")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,\
        help="whether to clip the value loss")
    parser.add_argument("--clip-coef", type=float, default=0.2, nargs="?", const=True,
        help="the surrogate clipping coefficient")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--vf-coef", type=float, default=0.5, nargs="?", const=True,\
        help="coeff for value loss contribution to total loss")
    parser.add_argument("--ent-coef", type=float, default=0.00, nargs="?", const=True,\
        help="entropy coefficient to encourage exploration")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,\
        help="whether to anneal the learning rate over the whole run")
    
    args = parser.parse_args()
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CombinedAgent(nn.Module):
    def __init__(self, env):
        super(CombinedAgent, self).__init__()
        self.n_layers = 128
        self.linear1 = layer_init(
            nn.Linear(env.observation_space.shape[0], self.n_layers)
        )
        self.activation1 = nn.ReLU()

        self.linear2 = layer_init(nn.Linear(self.n_layers, self.n_layers))
        self.activation2 = nn.ReLU()

        self.mean_action = nn.Linear(self.n_layers, env.action_space.shape[0])

        self.sigma_action = nn.Linear(self.n_layers, env.action_space.shape[0])
        self.sigma_activation = nn.Sigmoid()

        self.critic_output = nn.Linear(self.n_layers, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)

        mean_action = self.mean_action(x)

        sigma = self.sigma_action(x)
        sigma_action = self.sigma_activation(sigma)

        value = self.critic_output(x)
        return mean_action, sigma_action, value


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")

    ### Seeding ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.env_id not in ["arthur_env", "resonator_env"]:
        raise ValueError("env id can only be arthur_env or resonator_env")
    env = ResonatorEnv()
    next_obs, info = env.reset(seed=args.seed)

    model = CombinedAgent(env).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    summary(model, (args.batch_size, env.observation_space.shape[0]))

    # ALGO Logic: Storage setup
    sample_obs = torch.zeros(
        (args.batch_size,) + env.observation_space.shape, requires_grad=False
    ).to(device)
    train_obs = torch.zeros(
        (args.batch_size,) + env.observation_space.shape, requires_grad=True
    ).to(device)

    # Start the Environment
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    for update in tqdm(range(1, args.num_updates + 1)):
        ### Use only what Arthur Includes, but in the CleanRL style ###
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            mean_action, sigma_action, critic_value = model(sample_obs)
            probs = Normal(mean_action, sigma_action)
            action = probs.sample()
            logprob = probs.log_prob(action).sum(1)

            temp_obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()
            )
            torch_obs = torch.tensor(temp_obs, dtype=torch.float32).to(device)
            rewards = torch.tensor(reward, dtype=torch.float32).to(device)

            advantages = rewards - critic_value

        # flatten the batch
        b_obs = torch_obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprob.reshape(-1)
        b_actions = action.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = rewards.reshape(-1)
        b_values = critic_value.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_mean, new_sigma, new_value = model(b_obs[mb_inds])
                probs = Normal(new_mean, new_sigma)

                entropy = probs.entropy().sum(1)
                new_logprob = probs.log_prob(b_actions[mb_inds]).sum(1)

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantage = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantage = (mb_advantage - mb_advantage.mean()) / (
                        mb_advantage.std() + 1e-8
                    )

                # Policy Loss
                pg_loss1 = -mb_advantage * ratio
                pg_loss2 = -mb_advantage * torch.clamp(
                    ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = new_value.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_mean_reward = np.asarray(info["mean reward"])
        batch_mean_max_photon = np.asarray(info["mean max photon"])
        batch_mean_max_separation = np.asarray(info["mean max separation"])

        if args.track:
            writer.add_scalar("charts/mean_reward", batch_mean_reward, update)
            writer.add_scalar("charts/mean_max_photon", batch_mean_max_photon, update)
            writer.add_scalar(
                "charts/mean_max_separation", batch_mean_max_separation, update
            )
            writer.add_scalar(
                "charts/advantage", advantages.detach().mean().numpy(), update
            )
            writer.add_scalar(
                "charts/epoch_time", int(update / (time.time() - start_time)), update
            )

            writer.add_scalar(
                "losses/critic_loss", v_loss.detach().mean().numpy(), update
            )
            writer.add_scalar(
                "losses/actor_loss", pg_loss.detach().mean().numpy(), update
            )
            writer.add_scalar(
                "losses/unclipped_actor_loss", pg_loss1.detach().mean().numpy(), update
            )
            writer.add_scalar("losses/total_loss", loss.detach().mean().numpy(), update)

        if args.print_debug:
            max_reward_obtained = info["max reward"]
            separation_at_max_reward = info["separation at max reward"]
            photon_at_max_reward = info["photon at max reward"]
            bandwidth_at_max_reward = info["bandwidth at max reward"]
            print("\n Update", update)
            print("Mean Batch Reward", batch_mean_reward)
            print("Mean Batch Max Separation", batch_mean_max_separation)
            print("Mean Batch Max Photon", batch_mean_max_photon)
            print("Max Reward Obtained", max_reward_obtained)
            print(f"Separation at Max Reward: {separation_at_max_reward}")
            print(f"Photon at Max Reward: {photon_at_max_reward}")
            print(f"Bandwidth at Max Reward: {bandwidth_at_max_reward}")

        if update % 100 == 0:
            action_at_max_reward = info["action at max reward"]
            res_state_at_max_reward = info["res state at max reward"]
            print(f"Action at max reward: {action_at_max_reward}")
            print(f"Res State at max reward: {res_state_at_max_reward}")

    if args.track:
        writer.close()
