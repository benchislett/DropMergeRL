import copy
from enum import Enum
import numpy as np
import pygame

from components.tile import tile_width_px, background_color, draw_tile
from components.grid import tile_padding_width_px, draw_grid
from components.simulator import DropMergeEnv

import torch
from stable_baselines3 import PPO


class Strategy(Enum):
    HUMAN = "human"
    RANDOM = "random"
    CLEVER = "clever"
    PPO = "ppo"
    PPO_LOOKAHEAD = "ppo_lookahead"

# ppo_path = "models/ppo_dropmerge.zip"
ppo_path = "model_checkpoints/rl_model_998400000_steps.zip"

def best_action_with_lookahead(model, env, depth: int = 1, gamma: float = 0.995):
    """
    Return the column index chosen by depth-`depth` stochastic expectimax.
    `model` must be a trained SB3 PPO object.
    """

    # ------------------------------------------------------------------ #
    # Recursive expectimax: chance nodes are the random next-tile spawn. #
    # ------------------------------------------------------------------ #
    def expectimax(env_state: DropMergeEnv, d: int) -> float:
        # leaf node: use value network
        if d == 0:
            obs_dict = env_state._observation()
            obs_tensor, _ = model.policy.obs_to_tensor(obs_dict)
            with torch.no_grad():
                return float(model.policy.predict_values(obs_tensor))

        # maximise over legal actions
        best_q = -float("inf")
        for a in env_state._legal_actions():
            env_after_action = copy.deepcopy(env_state)
            obs, r, term, trunc, _ = env_after_action.step(a)

            # terminal state -> no further look-ahead
            if term or trunc:
                exp_val = 0.0
            else:
                # ---------------- Chance node ----------------
                child_states = []
                for tile in DropMergeEnv.SPAWNABLE_VALUES:
                    env_child = copy.deepcopy(env_after_action)
                    env_child.current_tile = tile
                    child_states.append(env_child)

                # batch-evaluate all six children
                obs_batch      = [s._observation() for s in child_states]
                obs_tensor  = [model.policy.obs_to_tensor(x)[0] for x in obs_batch]
                with torch.no_grad():
                    v_batch = [float(model.policy.predict_values(x)) for x in obs_tensor]

                exp_val = float(np.mean(v_batch))  # uniform spawn distribution

            q = r + gamma * exp_val
            best_q = max(best_q, q)

        return best_q

    # ---------------- Root search ----------------
    legal_actions = env._legal_actions()
    q_values = []
    for a in legal_actions:
        env_root_child = copy.deepcopy(env)
        _, r, term, trunc, _ = env_root_child.step(a)

        if term or trunc:
            exp_val = 0.0
        else:
            exp_val = 0.0
            for tile in DropMergeEnv.SPAWNABLE_VALUES:
                env_child = copy.deepcopy(env_root_child)
                env_child.current_tile = tile
                exp_val += (1.0 / len(DropMergeEnv.SPAWNABLE_VALUES)) * expectimax(
                    env_child, depth - 1
                )

        q_values.append(r + gamma * exp_val)

    # Choose arg-max action
    best_idx = int(np.argmax(q_values))
    return legal_actions[best_idx]

if __name__ == "__main__":
    num_rows = 7
    num_cols = 5

    play_strategy = Strategy.PPO_LOOKAHEAD
    if play_strategy == Strategy.PPO or play_strategy == Strategy.PPO_LOOKAHEAD:
        assert ppo_path is not None, "PPO path must be provided for PPO strategy."
        if torch.cuda.is_available():
            model = PPO.load(ppo_path, device='cuda')
        else:
            model = PPO.load(ppo_path, device='cpu')
            print("WARNING: Running PPO strategy on CPU, GPU not detected.")

    grid_total_width = num_cols * tile_width_px + (num_cols + 1) * tile_padding_width_px
    grid_total_height = num_rows * tile_width_px + (num_rows + 1) * tile_padding_width_px

    top_ui_height = 10 * 2 + tile_width_px

    window_total_width = grid_total_width
    window_total_height = grid_total_height + top_ui_height

    pygame.init()
    env = DropMergeEnv(seed=42)
    obs, _ = env.reset()
    delta_captures = []

    screen = pygame.display.set_mode((window_total_width, window_total_height))
    pygame.display.set_caption("Drop Merge Game")

    clock = pygame.time.Clock()

    selected_col = None

    display_queue = []
    merge_indicators = []

    game_over = False

    num_moves = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            if not display_queue and play_strategy == Strategy.HUMAN:
                # If display queue is non-empty, ignore user input and continue playing the animation

                # Handle key presses for actions
                # mouse hover selects column
                if event.type == pygame.MOUSEMOTION:
                    mx, _ = event.pos
                    selected_col = mx // (tile_width_px + tile_padding_width_px) if mx < grid_total_width else None
                
                # mouse click triggers placement step
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and selected_col is not None:
                    result, delta_captures = env._step(selected_col)
                    display_queue.extend(delta_captures)
                    if result == DropMergeEnv.StepResult.SUCCESS:
                        num_moves += 1
                        obs = env._observation()
                    else:
                        print("Invalid action or column full.")
                    selected_col = None
        
        if env.random_action() is None:
            game_over = True
            print(f"Game Over! Played {num_moves} moves! Resetting the environment.")
            num_moves = 0
            obs, _ = env.reset()
            display_queue.clear()
            selected_col = None
            continue

        if not display_queue and play_strategy != Strategy.HUMAN:
            if play_strategy == Strategy.RANDOM:
                # Randomly select a column to drop the tile
                selected_col = env.random_action()
            elif play_strategy == Strategy.CLEVER:
                legal_actions = env._legal_actions()
                board_after_each_action = [env._apply_step_no_mutate(col, env.board, env.current_tile)[0] for col in legal_actions]
                heuristic_scores = [env._heuristic_score(board) for board in board_after_each_action]
                best_action = legal_actions[heuristic_scores.index(max(heuristic_scores))]
                selected_col = best_action
            elif play_strategy == Strategy.PPO:
                action, _ = model.predict(obs, deterministic=True)
                selected_col = int(action.item() if isinstance(action, torch.Tensor) else action)
                legal_actions = env._legal_actions()
                if selected_col not in legal_actions:
                    selected_col = env.random.choice(legal_actions)
            elif play_strategy == Strategy.PPO_LOOKAHEAD:
                selected_col = best_action_with_lookahead(model, env, depth=2)
            result, delta_captures = env._step(selected_col)
            display_queue.extend(delta_captures)
            if result == DropMergeEnv.StepResult.SUCCESS:
                num_moves += 1
                obs = env._observation()
            else:
                print("Invalid action or column full.")
            selected_col = None

        screen.fill(background_color)

        # Draw top UI
        # Draw the current tile
        current_tile = env.current_tile
        next_tile = env.next_tile

        downscale = 1.5
        tile_width_small_px = int(tile_width_px / downscale)

        draw_tile(screen, env.current_tile, (window_total_width // 2 - (tile_width_px // 2), 10), aa_scale=4)
        draw_tile(screen, env.next_tile, (window_total_width // 2 + (tile_width_px // 2) + (tile_width_px - tile_width_small_px) // 2, 10 + (tile_width_px - tile_width_small_px) // 2), downscale=1.5, aa_scale=4)

        if display_queue:
            elem = display_queue.pop(0)
            board_capture, merge_indicators = elem
            draw_grid(screen, (0, top_ui_height), board_capture, selected_col, env.current_tile, merge_indicators)
            pygame.display.flip()
            # add a delay to see the animation
            # skipping the last animation as it should be the final state
            if display_queue and play_strategy == Strategy.HUMAN:
                pygame.time.delay(250)
            elif display_queue:
                # pygame.time.delay(10)
                pass
        else:
            draw_grid(screen, (0, top_ui_height), env.board, selected_col, env.current_tile, merge_indicators)
            pygame.display.flip()
        # clock.tick(10)

