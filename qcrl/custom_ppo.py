import os
import argparse
import time
import random
from tqdm import tqdm
from distutils.util import strtobool

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from complexPyTorch.complexLayers import (
    ComplexLinear,
)
from complexPyTorch.complexFunctions import complex_relu, complex_tanh, complex_sigmoid

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import wandb

from readout_optimisation.rl_envs.gymnasium_env import ReadoutEnv

# For debugging
# torch.autograd.set_detect_anomaly(True)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="Complex-valued PPO",
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
    parser.add_argument("--dtype-bits", type=int, default=32, nargs="?", const=True,
        help="The byte size that will be used for float and complex dtypes")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="readout_env",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1, nargs="?", const=True,\
        help="number of environments for parallel processing")
    parser.add_argument("--num-updates", type=int, default=4000, nargs="?", const=True,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0003, nargs="?", const=True,
        help="the learning rate of the optimizer")
    parser.add_argument("--batch-size", type=int, default=100, nargs="?", const=True,\
        help="batch size for each update")
    parser.add_argument("--clip-epsilon", type=float, default=0.1, nargs="?", const=True,\
        help="clipping epsilon")
    parser.add_argument("--grad-clip", type=float, default=0.5, nargs="?", const=True,\
        help="gradient clip for optimizer updates")
    parser.add_argument("--vf-coeff", type=float, default=0.5, nargs="?", const=True,\
        help="coeff for value loss contribution to total loss")
    parser.add_argument("--ent-coeff", type=float, default=0.01, nargs="?", const=True,\
        help="entropy coefficient to encourage exploration")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,\
        help="whether to anneal the learning rate over the whole run")
    
    args = parser.parse_args()
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Need to handle Complex Layer Custom Initialisation
class CombinedAgent(nn.Module):
    def __init__(self, env):
        super(CombinedAgent, self).__init__()
        self.n_layers = 128
        self.linear1 = ComplexLinear(env.observation_space.shape[0], self.n_layers)
        # self.activation1 = ComplexReLU()

        self.linear2 = ComplexLinear(self.n_layers, self.n_layers)
        # self.activation2 = ComplexReLU()

        self.mean_action = ComplexLinear(self.n_layers, env.action_space.shape[0])

        self.sigma_action = ComplexLinear(self.n_layers, env.action_space.shape[0])
        # self.sigma_activation = ComplexSigmoid()

        self.critic_output = ComplexLinear(self.n_layers, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = complex_relu(x)

        x = self.linear2(x)
        x = complex_relu(x)

        mean_action = self.mean_action(x)

        sigma = self.sigma_action(x)
        sigma_action = complex_sigmoid(sigma)

        value = self.critic_output(x)
        return mean_action, sigma_action, value


if __name__ == "__main__":
    args = parse_args()
    if args.dtype_bits == 32:
        float_dtype = torch.float32
        complex_dtype = torch.cfloat
    elif args.dtype_bits == 64:
        float_dtype = torch.float64
        complex_dtype = torch.cdouble
    else:
        raise ValueError("For now we only support 32 or 64 bit dtypes")
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

    assert args.env_id == "readout_env", f"env_id must be readout_env"
    env = ReadoutEnv()
    next_obs, info = env.reset(seed=args.seed)

    model = CombinedAgent(env).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"nn model: {model}")

    # ALGO Logic: Storage setup
    sample_obs = torch.zeros(
        env.observation_space.shape,
        requires_grad=False,
        dtype=complex_dtype,
    ).to(device)
    train_obs = torch.zeros(
        env.observation_space.shape,
        requires_grad=True,
        dtype=complex_dtype,
    ).to(device)

    # Start the Environment
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    for update in tqdm(range(1, args.num_updates + 1)):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            mean_action, sigma_action, critic_value = model(sample_obs)
            real_action = torch.view_as_real(mean_action)
            real_sigma = torch.view_as_real(sigma_action)
            probs = Normal(real_action, real_sigma)
            action = probs.sample()
            logprob = probs.log_prob(action).sum(1)
            action_complex = torch.view_as_complex(action)
            print(f"probs: {probs}")
            print(f"logprob: {logprob}, shape: {logprob.shape}")
            print(f"action complex: {action_complex}, shape: {action_complex.shape}")

            temp_obs, reward, terminated, truncated, info = env.step(
                action_complex.cpu().numpy()
            )
            rewards = torch.tensor(reward).to(device)

            advantages = rewards - critic_value

        new_mean, new_sigma, new_value = model(train_obs)
        new_logprob = Normal(new_mean, new_sigma).log_prob(action).sum(1)
        log_ratio = new_logprob - logprob
        ratio = log_ratio.exp()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = ((new_value - rewards) ** 2).mean()
        loss = pg_loss + v_loss * args.vf_coeff

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if args.track:
            writer.add_scalar("charts/mean_reward", info["mean rewards"], update)
            writer.add_scalar(
                "charts/average_fidelity", info["average fidelity"], update
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
            print("\n Update", update)
            print("Average reward", info["mean rewards"])
            print("Average Gate Fidelity:", info["average fidelity"])
            print("Max Reward", info["max reward"])
            print("Max Fidelity", info["max fidelity"])

    writer.close()
