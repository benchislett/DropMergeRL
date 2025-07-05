from abc import ABC
import copy
from enum import Enum
import numpy as np
import pygame
from tqdm import tqdm

from components.tile import tile_width_px, background_color, draw_tile
from components.grid import tile_padding_width_px, draw_grid
from components.simulator import DropMergeEnv

import torch
from stable_baselines3 import PPO

class ValueFunction(ABC):
    def get_value(self, board: np.ndarray, current_tile: int | None, next_tile: int | None) -> float:
        """
        Get the value of the board state.
        This is a placeholder for the actual implementation.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_value_batched(self, boards: list[np.ndarray], current_tiles: list[int] | None, next_tiles: list[int] | None) -> list[float]:
        """
        Get the values of multiple board states in a batch.
        This is a reference implementation, but can be optimized for some use cases.
        """
        if current_tiles and next_tiles:
            return [self.get_value(board, current_tile, next_tile) for board, current_tile, next_tile in zip(boards, current_tiles, next_tiles)]
        else:
            return [self.get_value(board, None, None) for board in boards]
    
    @staticmethod
    def is_board_only() -> bool:
        """
        Check if the value function only considers the board state.
        This is a placeholder for the actual implementation.
        """
        return False

class Agent(ABC):
    """
    Base class for agents that can play the Drop Merge game.
    """
    def get_action(self, env: DropMergeEnv):
        """
        Select an action based on the agent's strategy.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class RandomAgent(Agent):
    """
    Agent that selects actions randomly.
    """
    def get_action(self, env: DropMergeEnv):
        return env.random_action() or 0

class ValueOptimizerAgent(Agent):
    """
    Agent that uses a given value function to select the best action.
    """
    def __init__(self, value_function: ValueFunction, depth: int = 1, **kwargs):
        self.value_function = value_function
        self.depth = depth
        self.env_playground = DropMergeEnv(**kwargs)

    def get_action_old(self, env: DropMergeEnv):
        assert self.depth == 1, "ValueOptimizerAgent only supports depth 1 for now."
        legal_actions = env._legal_actions()
        if not legal_actions:
            return env.random_action() or 0
        
        num_tiles = len(DropMergeEnv.SPAWNABLE_VALUES)
        boards_batch = []
        next_tiles_batch = []
        spawn_values_batch = []
        for action in legal_actions:
            tmp_board, _, _ = env._apply_step_no_mutate(action, env.board, env.current_tile)
            boards_batch.extend([tmp_board] * num_tiles)
            next_tiles_batch.extend([env.next_tile] * num_tiles)
            spawn_values_batch.extend(DropMergeEnv.SPAWNABLE_VALUES)

        batched_values = self.value_function.get_value_batched(
            boards_batch, next_tiles_batch, spawn_values_batch
        )

        best_action = None
        best_value = -float("inf")
        for i, action in enumerate(legal_actions):
            start = i * num_tiles
            value = sum(batched_values[start : start + num_tiles]) / float(num_tiles)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action if best_action is not None else (env.random_action() or 0)
    
    def get_action(self, env: DropMergeEnv):
        assert self.depth >= 1, "ValueOptimizerAgent requires depth ≥ 1."
        legal_actions = env._legal_actions()
        if not legal_actions:
            return env.random_action() or 0

        orig_board, orig_cur, orig_next = env.board, env.current_tile, env.next_tile
        spawnables = DropMergeEnv.SPAWNABLE_VALUES
        boards_batch, next_tiles_batch, spawn_values_batch = [], [], []

        class _N:
            __slots__ = ("k", "c", "i")
            def __init__(self, k, c=None, i=None):
                self.k, self.c, self.i = k, c, i
            def v(self, leaf):
                if self.k == 0:
                    return leaf[self.i] if self.i is not None else -float("inf")
                if self.k == 1:
                    return sum(n.v(leaf) for n in self.c) / len(self.c)
                best = -float("inf")
                for n in self.c:
                    v = n.v(leaf)
                    if v > best:
                        best = v
                return best

        def _legal(b):
            old = env.board
            env.board = b
            acts = env._legal_actions()
            env.board = old
            return acts

        def chance_node(b, cur, d):
            ch = []
            for sv in spawnables:
                if d == 0:
                    idx = len(boards_batch)
                    boards_batch.append(b)
                    next_tiles_batch.append(cur)
                    spawn_values_batch.append(sv)
                    ch.append(_N(0, i=idx))
                else:
                    ch.append(decision_node(b, cur, sv, d))
            return _N(1, ch)

        def decision_node(b, cur, nxt, d):
            acts = _legal(b)
            if not acts:
                return _N(0, i=None)
            return _N(2, [chance_node(env._apply_step_no_mutate(a, b, cur)[0], nxt, d - 1) for a in acts])

        roots = []
        for a in legal_actions:
            nb = env._apply_step_no_mutate(a, orig_board, orig_cur)[0]
            roots.append((a, chance_node(nb, orig_next, self.depth - 1)))

        if not boards_batch:
            env.board, env.current_tile, env.next_tile = orig_board, orig_cur, orig_next
            return env.random_action() or 0

        assert len(boards_batch) == len(next_tiles_batch) == len(spawn_values_batch)

        leaf_vals = self.value_function.get_value_batched(
            boards_batch, next_tiles_batch, spawn_values_batch
        )

        best_val, best_act = -float("inf"), None
        for a, n in roots:
            v = n.v(leaf_vals)
            if v > best_val:
                best_val, best_act = v, a

        env.board, env.current_tile, env.next_tile = orig_board, orig_cur, orig_next
        return best_act if best_act is not None else env.random_action() or 0

def pad_observation(obs, expected_shape):
    """
    Pad the observation to match the expected shape.
    """
    actual_shape = obs['board'].shape
    if actual_shape[0] < expected_shape[0]:
        padding = np.zeros((expected_shape[0] - actual_shape[0], *actual_shape[1:]), dtype=obs['board'].dtype)
        obs['board'] = np.concatenate((obs['board'], padding), axis=0)
    return obs

class PPOAgent(Agent):
    """
    Agent that uses a PPO model to select the best action.
    """
    def __init__(self, model_path: str):
        model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    @torch.inference_mode()
    def get_action(self, env: DropMergeEnv):
        obs = env._observation()
        obs = pad_observation(obs, self.model.observation_space['board'].shape)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action.item() if isinstance(action, torch.Tensor) else action)
        if action not in env._legal_actions():
            action = env.random_action() or 0
        return action

class PPOValueFunction(ValueFunction):
    """
    Value function that uses a PPO model to estimate the value of a board state.
    """
    def __init__(self, model_path: str, **kwargs):
        self.model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.env_playground = DropMergeEnv(**kwargs)

    @torch.inference_mode()
    def get_value(self, board: np.ndarray, current_tile: int, next_tile: int) -> float:
        self.env_playground.board = board
        self.env_playground.current_tile = current_tile
        self.env_playground.next_tile = next_tile
        obs = self.env_playground._observation()
        obs = pad_observation(obs, self.model.observation_space['board'].shape)
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        return float(self.model.policy.predict_values(obs_tensor))
    
    @torch.inference_mode()
    def get_value_batched(self, boards, current_tiles, next_tiles):
        # collect observations
        obs_batch = {}
        for board, current_tile, next_tile in zip(boards, current_tiles, next_tiles):
            self.env_playground.board = board
            self.env_playground.current_tile = current_tile
            self.env_playground.next_tile = next_tile
            new_obs = self.env_playground._observation()
            new_obs = pad_observation(new_obs, self.model.observation_space['board'].shape)
            for k in new_obs:
                expanded_val = np.expand_dims(new_obs[k], axis=0)
                if k in obs_batch:
                    obs_batch[k] = np.concat((obs_batch[k], expanded_val), axis=0)
                else:
                    obs_batch[k] = expanded_val
        obs_tensor_dict = self.model.policy.obs_to_tensor(obs_batch)[0]
        v_batch = self.model.policy.predict_values(obs_tensor_dict)
        return [float(v) for v in v_batch]

    def is_board_only(self) -> bool:
        return False
    
class CustomValueFunction(ValueFunction):
    def get_value(self, board: np.ndarray, _current_tile: int | None, _next_tile: int | None) -> float:
        """
        Custom value function that performs analysis based on the board state.
        Components:
        - Accumulate the sum of all tile values on the board. This is the "base score", encouraging multi-way merges.
        - Look for "trapped" tiles that are surrounded by higher tiles, and are "stuck". These should be penalized heavily.
        - Apply bonuses based on tiles that are above their successor in the merge chain, such as a "32" above a "64".
        """
        # base_score = np.sum(board)
        base_score = 0

        # Subtract 1 for each nonzero tile to encourage more merges
        base_score -= np.count_nonzero(board)

        # Punish non-decreasing columns
        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 1, 0, -1):
                if board[row, col] > 0 and board[row, col] < board[row - 1, col]:
                    base_score -= 100
                    break

        top_row_last_values = []
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] > 0:
                    top_row_last_values.append(board[row, col])
                    break
            else:
                top_row_last_values.append(0)
        
        top_row_last_values_no_zeros = [v for v in top_row_last_values if v > 0]
        delta = len(top_row_last_values_no_zeros) - len(set(top_row_last_values_no_zeros))
        if delta > 0:
            base_score -= delta

        return base_score

    def is_board_only(self) -> bool:
        return True

if __name__ == "__main__":
    num_rows = 7
    num_cols = 5
    env = DropMergeEnv(num_rows=num_rows, num_cols=num_cols, seed=44)

    # agent = PPOAgent("preav4.zip")
    # agent = RandomAgent()
    # agent = ValueOptimizerAgent(PPOValueFunction("preav4.zip", num_rows=num_rows, num_cols=num_cols), depth=3, num_rows=num_rows, num_cols=num_cols)
    agent = ValueOptimizerAgent(CustomValueFunction(), depth=2, num_rows=num_rows, num_cols=num_cols)

    num_sims = 100
    stats_per_run = []

    for i in tqdm(range(num_sims)):
        obs, _ = env.reset()
        game_over = False
        num_steps = 0
        while not game_over:
            action = agent.get_action(env)
            result, _ = env._step(action)
            if result == DropMergeEnv.StepResult.SUCCESS:
                obs = env._observation()
            else:
                game_over = True
            num_steps += 1
            if num_steps % 100 == 0:
                print(f"Simulation {i + 1}/{num_sims}, steps: {num_steps}, board sum: {np.sum(env.board)}")
        board_sum = np.sum(env.board)
        stats_per_run.append((num_steps, board_sum))
        print(np.mean([s[0] for s in stats_per_run]), np.mean([s[1] for s in stats_per_run]))
    
    avg_steps = np.mean([s[0] for s in stats_per_run])
    avg_board_sum = np.mean([s[1] for s in stats_per_run])

    p5_steps = np.percentile([s[0] for s in stats_per_run], 5)
    p25_steps = np.percentile([s[0] for s in stats_per_run], 25)
    p75_steps = np.percentile([s[0] for s in stats_per_run], 75)
    p95_steps = np.percentile([s[0] for s in stats_per_run], 95)

    print(f"Average steps: {avg_steps:.1f} ± {np.std([s[0] for s in stats_per_run]):.2f}")
    
    print()
    print(f"5th percentile steps: {p5_steps:.1f}")
    print(f"25th percentile steps: {p25_steps:.1f}")
    print(f"50th percentile steps: {avg_steps:.1f}")
    print(f"75th percentile steps: {p75_steps:.1f}")
    print(f"95th percentile steps: {p95_steps:.1f}")
    print()

    print(f"Average board sum: {avg_board_sum:.1f}")
