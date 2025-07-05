import copy
from enum import Enum
import numpy as np
import pygame

from components.tile import tile_width_px, background_color, draw_tile
from components.grid import tile_padding_width_px, draw_grid
from components.simulator import DropMergeEnv
from components.strategy import CustomValueFunction, PPOValueFunction, RandomAgent, ValueOptimizerAgent, PPOAgent

import torch
from stable_baselines3 import PPO

if __name__ == "__main__":
    num_rows = 4
    num_cols = 5

    # agent = PPOAgent("preav5_4x5.zip")
    # agent = RandomAgent()
    # agent = ValueOptimizerAgent(PPOValueFunction("preav3.zip"), depth=1)
    # agent = ValueOptimizerAgent(CustomValueFunction(), depth=1)
    agent = None
    fast_replay = False

    grid_total_width = num_cols * tile_width_px + (num_cols + 1) * tile_padding_width_px
    grid_total_height = num_rows * tile_width_px + (num_rows + 1) * tile_padding_width_px

    top_ui_height = 10 * 2 + tile_width_px

    window_total_width = grid_total_width
    window_total_height = grid_total_height + top_ui_height

    pygame.init()
    env = DropMergeEnv(num_rows=num_rows, num_cols=num_cols, seed=42)
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
            
            if not display_queue and agent is None:
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

        if not display_queue and agent is not None:
            selected_col = agent.get_action(env)
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
            if display_queue and not fast_replay:
                pygame.time.delay(250)
            elif display_queue:
                pygame.time.delay(10)
        else:
            draw_grid(screen, (0, top_ui_height), env.board, selected_col, env.current_tile, merge_indicators)
            pygame.display.flip()
        if fast_replay or agent is None:
            clock.tick(60)
        elif not display_queue:
            clock.tick(1)
        else:
            clock.tick(10)

