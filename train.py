#!/usr/bin/env python
"""Basic PPO trainer for the Drop‑Merge game.

Usage
-----
$ python train_dropmerge_rl.py \
        --total-timesteps 1e8 \
        --n-envs 16 \
        --log-dir runs/ppo_dropmerge \
        --save-path models/ppo_dropmerge

The script records the following key metrics via Stable‑Baselines3’s
logger (TensorBoard‑compatible):
    rollout/ep_rew_mean   – average episode (game) reward
    rollout/ep_len_mean   – average episode length (number of steps)
Both are computed on‑policy during training and averaged over the last
100 completed episodes.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Import the game environment
from components.simulator import DropMergeEnv
from components.strategy import pad_observation

###############################################################################
# Observation pre‑processing wrapper                                          #
###############################################################################


class BoolToFloat32(gym.ObservationWrapper):
    """Cast all boolean observation planes to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Clone the original spaces but change the dtype to float32 so that
        # Stable‑Baselines3 does not complain about unexpected dtypes.
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=space.shape,
                dtype=np.float32,
            )
            for k, space in env.observation_space.items()
        })
        self.observation_space['board'] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7, 5, self.observation_space['board'].shape[2]),
            dtype=np.float32,
        )

    def observation(self, obs):  # type: ignore[override]
        obs = pad_observation(obs, self.observation_space['board'].shape)
        return {
            "board": obs["board"].astype(np.float32),
            "current_tile": obs["current_tile"].astype(np.float32),
            "next_tile": obs["next_tile"].astype(np.float32),
        }

###############################################################################
# Custom feature extractor (CNN + MLP)                                        #
###############################################################################


class DropMergeFeatureExtractor(BaseFeaturesExtractor):
    """Implements the architecture discussed in the design doc."""

    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)  # temp

        # ── Board branch ────────────────────────────────────────────────────
        board_shape = observation_space["board"].shape
        channels_last = board_shape[2]
        self.board_net = nn.Sequential(
            nn.Conv2d(channels_last, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # ── Tile branch ─────────────────────────────────────────────────────
        tile_len = observation_space["current_tile"].shape[0] + observation_space["next_tile"].shape[0]
        self.tile_net = nn.Sequential(
            nn.Linear(tile_len, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Final feature dimension
        self._features_dim = 64 + 32

    def forward(self, obs):  # type: ignore[override]
        # `obs` is a Dict with keys as tensors
        board = obs["board"].permute(0, 3, 1, 2)  # (B, C, H, W)

        board_feat = self.board_net(board)

        tiles = torch.cat([obs["current_tile"], obs["next_tile"]], dim=1)
        tile_feat = self.tile_net(tiles)

        return torch.cat([board_feat, tile_feat], dim=1)

###############################################################################
# Custom callback to log episode length                                       #
###############################################################################


class EpLenRewardCallback(BaseCallback):
    """Logs average episode length and reward every 100 finished episodes."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_lengths: list[int] = []
        self.ep_rewards: list[float] = []

    def _on_step(self) -> bool:  # noqa: D401, N802
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_rewards.append(info["episode"]["r"])
        if len(self.ep_lengths) >= 100:
            self.logger.record("rollout/ep_len_mean", float(np.mean(self.ep_lengths)))
            self.logger.record("rollout/ep_rew_mean", float(np.mean(self.ep_rewards)))
            self.ep_lengths.clear()
            self.ep_rewards.clear()
        return True

###############################################################################
# Utility: environment factory                                               #
###############################################################################


def make_single_env(seed: int | None = None) -> Callable[[], gym.Env]:
    """Returns a thunk that creates one wrapped environment."""

    def _init() -> gym.Env:
        env = DropMergeEnv(num_rows=7, num_cols=5, seed=seed)
        env = BoolToFloat32(env)
        return env

    return _init

###############################################################################
# Main training routine                                                      #
###############################################################################

def warmup_then_decay(
    base_lr     = 1e-4,   # start
    peak_lr     = 1e-3,   # max after warm-up
    warmup_frac = 0.03,   # first 3 % of total steps
    final_lr    = 5e-5    # end of run
):
    """
    Returns a schedule callable for SB3:
        progress_remaining = 1.0 ... 0.0
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining            # 0 → 1
        if progress < warmup_frac:
            # linear warm-up
            return base_lr + (peak_lr - base_lr) * (progress / warmup_frac)
        else:
            # linear decay
            decay_progress = (progress - warmup_frac) / (1.0 - warmup_frac)
            return peak_lr + (final_lr - peak_lr) * decay_progress
    return schedule

def linear_schedule(
    start: float,
    end: float):
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return start + (end - start) * progress
    return schedule

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train PPO on Drop‑Merge")
    parser.add_argument("--total-timesteps", type=int, default=int(1e8))
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--log-dir", type=str, default="runs/ppo_dropmerge")
    parser.add_argument("--save-path", type=str, default="models/ppo_dropmerge")
    args = parser.parse_args()

    # Vectorised environment with monitoring
    env = make_vec_env(
        make_single_env(),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    env = VecMonitor(env, filename=args.log_dir)

    policy_kwargs = dict(
        features_extractor_class=DropMergeFeatureExtractor,
    )

    model = PPO.load("model_checkpoints/rl_model_2889600000_steps.zip", tensorboard_log=args.log_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    model.set_env(env)

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     policy_kwargs=policy_kwargs,
    #     n_steps=1024,
    #     batch_size=1024,
    #     n_epochs=1,
    #     learning_rate=warmup_then_decay(1e-5, 5e-4, 0.0, 1e-5),
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     verbose=1,
    #     tensorboard_log=args.log_dir,
    # )

    callback = EpLenRewardCallback()

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./model_checkpoints/',
                                         name_prefix='rl_model')

    callback = CallbackList([checkpoint_callback, callback])

    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)

    # Persist the trained network parameters
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
