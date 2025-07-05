from enum import Enum
import random
import time
from typing import List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import cpp_core

class DropMergeEnv(gym.Env):

    SPAWNABLE_VALUES = (2, 4, 8, 16, 32, 64)
    VALID_VALUES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    MAX_VALID_VALUE = 16384

    class StepResult(Enum):
        FAIL_INVALID = 0
        FAIL_FULL = 1
        SUCCESS = 2
        

    def __init__(self, num_rows = 7, num_cols = 5, seed: int = 42) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.random = random.Random(seed)

        self.action_space = spaces.Discrete(self.num_cols)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.num_rows, self.num_cols, len(self.VALID_VALUES)), dtype=bool),
            "current_tile": spaces.Box(low=0, high=1, shape=(len(self.SPAWNABLE_VALUES),), dtype=bool),
            "next_tile": spaces.Box(low=0, high=1, shape=(len(self.SPAWNABLE_VALUES),), dtype=bool),
        })
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.current_tile = self.next_tile = 0
        self.reset()
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_tile = self._random_tile()
        self.next_tile = self._random_tile()
        return self._observation(), {}
    
    def random_action(self) -> int | None:
        """Returns a random valid action (column index)."""
        legal_actions = self._legal_actions()
        if not legal_actions:
            return None
        
        return self.random.choice(legal_actions)
    
    def _legal_actions(self) -> List[int]:
        """Returns a list of legal actions (column indices)."""
        return [c for c in range(self.num_cols) if self._is_legal_action(c)]

    def _is_legal_action(self, action: int) -> bool:
        """Checks if the action (column index) is valid."""
        if action < 0 or action >= self.num_cols:
            return False
        
        if self.board[0, action] == 0:
            return True
        
        if self.board[0, action] == self.current_tile and self.board[0, action] < self.MAX_VALID_VALUE:
            return True
        
        return False
    
    def _resolve_board(self, board: np.ndarray, selected_col: int) -> np.ndarray:
        """Resolves the board state by applying gravity and merging tiles. Does not modify the original board, or `self`."""
        board = board.copy()
        delta_captures = []
        found_any_merge = True
        while found_any_merge:
            found_any_merge = []

            # Look for merges
            delta_captures_addition = None
            found_four_way_merge = None
            found_three_way_l_merge = None
            found_two_way_vertical_merge = None
            found_three_way_horizontal_merge = None
            found_two_way_horizontal_merge = None
            found_any_merge = False
            for r in range(self.num_rows):
                for c in range(self.num_cols):

                    if board[r, c] == 0 or board[r, c] >= self.MAX_VALID_VALUE:
                        continue
                    center = board[r, c]
                    left = board[r, c - 1] if c > 0 else 0
                    right = board[r, c + 1] if c < self.num_cols - 1 else 0
                    down = board[r + 1, c] if r < self.num_rows - 1 else 0

                    if found_four_way_merge is not None:
                        continue

                    if (left == center and right == center and down == center and center * 8 <= self.MAX_VALID_VALUE):
                        # First, check for four-way merges with three horizontal neighbours and one vertical neighbour
                        new_board = board.copy()
                        new_board[r + 1, c] = center * 8
                        new_board[r, c] = 0
                        new_board[r, c - 1] = 0
                        new_board[r, c + 1] = 0
                        found_four_way_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r + 1, c), (r, c)), ((r, c - 1), (r, c)), ((r, c + 1), (r, c))])
                        continue

                    if found_three_way_l_merge is not None:
                        continue

                    if (center == down and center == right and center * 4 <= self.MAX_VALID_VALUE):
                        # Check for three-way L-shaped merges with one vertical neighbour
                        new_board = board.copy()
                        new_board[r + 1, c] = center * 4
                        new_board[r, c] = 0
                        new_board[r, c + 1] = 0
                        found_three_way_l_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r + 1, c), (r, c)), ((r, c + 1), (r, c))])
                        continue

                    if found_two_way_vertical_merge is not None:
                        continue
                    if (center == down and center == left and center * 4 <= self.MAX_VALID_VALUE):
                        # Check for three-way L-shaped merges with one vertical neighbour
                        new_board = board.copy()
                        new_board[r + 1, c] = center * 4
                        new_board[r, c] = 0
                        new_board[r, c - 1] = 0
                        found_three_way_l_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r + 1, c), (r, c)), ((r, c - 1), (r, c))])
                        continue

                    if (center == down and center * 2 <= self.MAX_VALID_VALUE):
                        # Check for two-way vertical merges
                        new_board = board.copy()
                        new_board[r + 1, c] = center * 2
                        new_board[r, c] = 0
                        found_two_way_vertical_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r + 1, c), (r, c))])
                        continue

                    if found_three_way_horizontal_merge is not None:
                        continue

                    if (left == center and right == center and center * 4 <= self.MAX_VALID_VALUE):
                        # Check for three-way horizontal merges
                        new_board = board.copy()
                        new_board[r, c] = center * 4
                        new_board[r, c - 1] = 0
                        new_board[r, c + 1] = 0
                        found_three_way_horizontal_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r, c), (r, c - 1)), ((r, c + 1), (r, c))])
                        continue

                    if found_two_way_horizontal_merge is not None:
                        continue
                    if (left == center and center * 2 <= self.MAX_VALID_VALUE):
                        # Check for two-way horizontal merges
                        # Determine which way to merge so that the tile is moved towards the selected column
                        new_board = board.copy()
                        if c > selected_col:
                            # Merge 'center' into 'left'
                            new_board[r, c] = 0
                            new_board[r, c - 1] = center * 2
                        else:
                            # Merge 'left' into 'center'
                            new_board[r, c] = center * 2
                            new_board[r, c - 1] = 0
                        found_two_way_horizontal_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r, c), (r, c - 1))])
                        continue

                    if (center == right and center * 2 <= self.MAX_VALID_VALUE):
                        # Check for two-way horizontal merges
                        # Determine which way to merge so that the tile is moved towards the selected column
                        new_board = board.copy()
                        if c < selected_col:
                            # Merge 'center' into 'right'
                            new_board[r, c] = 0
                            new_board[r, c + 1] = center * 2
                        else:
                            # Merge 'right' into 'center'
                            new_board[r, c] = center * 2
                            new_board[r, c + 1] = 0
                        found_two_way_horizontal_merge = new_board
                        found_any_merge = True
                        delta_captures_addition = (new_board.copy(), [((r, c), (r, c + 1))])
                        continue
                pass
            if found_four_way_merge is not None:
                board = found_four_way_merge
            elif found_three_way_l_merge is not None:
                board = found_three_way_l_merge
            elif found_two_way_vertical_merge is not None:
                board = found_two_way_vertical_merge
            elif found_three_way_horizontal_merge is not None:
                board = found_three_way_horizontal_merge
            elif found_two_way_horizontal_merge is not None:
                board = found_two_way_horizontal_merge
            
            if found_any_merge:
                delta_captures.append(delta_captures_addition)

            # Apply gravity
            for c in range(self.num_cols):
                moved_anything = True
                while moved_anything:
                    moved_anything = False
                    for r in range(self.num_rows - 1, 0, -1):
                        if board[r, c] == 0 and board[r - 1, c] != 0:
                            # Move tile down
                            board[r, c] = board[r - 1, c]
                            board[r - 1, c] = 0
                            moved_anything = True
                            delta_captures.append((board.copy(), []))
        
        return board, delta_captures
    
    def _observation(self) -> Dict[str, Any]:
        board_encoded = np.zeros((self.num_rows, self.num_cols, len(self.VALID_VALUES)), dtype=np.bool)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.board[r, c] in self.VALID_VALUES:
                    value_index = self.VALID_VALUES.index(self.board[r, c])
                    board_encoded[r, c, value_index] = True
        
        current_tile_encoded = np.zeros(len(self.SPAWNABLE_VALUES), dtype=np.bool)
        if self.current_tile in self.SPAWNABLE_VALUES:
            current_tile_index = self.SPAWNABLE_VALUES.index(self.current_tile)
            current_tile_encoded[current_tile_index] = True

        next_tile_encoded = np.zeros(len(self.SPAWNABLE_VALUES), dtype=np.bool)
        if self.next_tile in self.SPAWNABLE_VALUES:
            next_tile_index = self.SPAWNABLE_VALUES.index(self.next_tile)
            next_tile_encoded[next_tile_index] = True

        return {
            "board": board_encoded,
            "current_tile": current_tile_encoded,
            "next_tile": next_tile_encoded,
        }

    def _random_tile(self) -> int:
        return self.random.choice(self.SPAWNABLE_VALUES)
    
    def _apply_step_no_mutate(self, action: int, board: np.ndarray, current_tile: int):
        """Performs a step without mutating the environment state."""
        if not 0 <= action < self.num_cols:
            return board, self.StepResult.FAIL_INVALID, []
        
        selected_col = action

        delta_captures = []
        # delta_captures.append((self.board.copy(), []))

        new_board = board.astype(np.int32).copy()
        success = cpp_core.step_inplace(new_board, selected_col, current_tile, self.MAX_VALID_VALUE)
        if not success:
            return new_board, self.StepResult.FAIL_FULL, []
        else:
            delta_captures.append((new_board.copy(), []))
            return new_board, self.StepResult.SUCCESS, delta_captures
    
    def _step(self, action: int):
        new_board, result, delta_captures = self._apply_step_no_mutate(action, self.board, self.current_tile)
        self.board = new_board.copy()
        self.current_tile = self.next_tile
        self.next_tile = self._random_tile()
        return result, delta_captures
    
    def _heuristic_score(self, board: np.ndarray) -> int:
        heuristic_score = 0
        for r in range(1, self.num_rows):
            for c in range(self.num_cols):
                if board[r, c] > 0:
                    if board[r - 1, c] > board[r, c]:
                        # If the tile above is larger, we incur a penalty proportional to the difference
                        heuristic_score -= (board[r - 1, c] - board[r, c])
                    elif board[r - 1, c] > 0 and board[r - 1, c] != board[r, c]:
                        # If the tile above is smaller, we gan a benefit proportional to the inverse difference:
                        # This encourages tiles to be stacked in descending order
                        heuristic_score += 128 / (board[r, c] - board[r - 1, c])

        return heuristic_score
    
    def step(self, action: int):
        score_before = self.board.sum()
        tile_before = self.current_tile
        heuristic_score_before = self._heuristic_score(self.board)
        result, delta_captures = self._step(action)
        heuristic_score_after = self._heuristic_score(self.board)
        score_after = self.board.sum()
        num_free_tiles = self.num_cols * self.num_rows - np.count_nonzero(self.board)

        # total_reward = max(0, score_after - score_before - tile_before) \
        #     + 0.5 * num_free_tiles \
        #     + (heuristic_penalty_after - heuristic_penalty_before)
        # total_reward = 1.0 + (heuristic_score_after - heuristic_score_before) / 128
        # num_full_rows = np.count_nonzero(np.all(self.board > 0, axis=1))
        # total_reward = (1.0 - (num_full_rows / (self.num_rows + 1))) / 128
        total_reward = 1.0 / 128

        if result == self.StepResult.SUCCESS:
            return self._observation(), total_reward, False, False, {"delta_captures": delta_captures}
        elif result == self.StepResult.FAIL_INVALID:
            raise ValueError(f"Invalid action: {action} is not a legal action.")
        elif result == self.StepResult.FAIL_FULL:
            # Game over, return a large negative reward
            return self._observation(), -10, True, False, {"delta_captures": delta_captures}
        else:
            raise ValueError(f"Unknown step result: {result}")
