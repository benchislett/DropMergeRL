import pygame

import numpy as np
from numpy.typing import NDArray

from components.tile import tile_width_px, background_color, draw_tile, TILE_COLORS

tile_padding_width_px = 1
grid_line_color = (0, 0, 0)

def draw_grid(surface: pygame.Surface, pos_offset: tuple[int, int], board: NDArray, selected_col: int | None = None, selected_col_value: int | None = None, merge_indicators: list[tuple[tuple[int, int], tuple[int, int]]] = []):
    num_rows, num_cols = board.shape

    total_width = num_cols * tile_width_px + (num_cols + 1) * tile_padding_width_px
    total_height = num_rows * tile_width_px + (num_rows + 1) * tile_padding_width_px

    grid_area_rect = pygame.Rect(pos_offset[0], pos_offset[1], total_width, total_height)
    surface.fill(background_color, grid_area_rect)

    # Draw grid lines
    for row in range(num_rows + 1):
        pygame.draw.line(
            surface,
            grid_line_color,
            (pos_offset[0], pos_offset[1] + row * (tile_width_px + tile_padding_width_px)),
            (pos_offset[0] + total_width, pos_offset[1] + row * (tile_width_px + tile_padding_width_px))
        )
    
    for col in range(num_cols + 1):
        pygame.draw.line(
            surface,
            grid_line_color,
            (pos_offset[0] + col * (tile_width_px + tile_padding_width_px), pos_offset[1]),
            (pos_offset[0] + col * (tile_width_px + tile_padding_width_px), pos_offset[1] + total_height)
        )
    
    # Draw a transluscent highlight for the selected column
    if selected_col is not None:
        if selected_col_value is None:
            raise ValueError("selected_col cannot be None if selected_col_value is None")
        if selected_col_value not in TILE_COLORS:
            raise ValueError(f"selected_col_value {selected_col_value} is not a valid tile value")
        highlight_rect = pygame.Rect(
            pos_offset[0] + selected_col * (tile_width_px + tile_padding_width_px) + tile_padding_width_px,
            pos_offset[1],
            tile_width_px,
            total_height
        )
        highlight_color = (*TILE_COLORS[selected_col_value], 128)
        highlight_surface = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        highlight_surface.fill(highlight_color)
        surface.blit(highlight_surface, highlight_rect.topleft)
    
    # Draw tiles
    for row in range(num_rows):
        for col in range(num_cols):
            value = board[row, col]
            if value > 0:
                pos_x = col * (tile_width_px + tile_padding_width_px) + tile_padding_width_px
                pos_y = row * (tile_width_px + tile_padding_width_px) + tile_padding_width_px
                draw_tile(surface, value, pos_offset=(pos_offset[0] + pos_x, pos_offset[1] + pos_y), aa_scale=4)

    # Draw merge indicators: connections from one tile to another
    for (tile_from, tile_to) in merge_indicators:
        from_row, from_col = tile_from
        to_row, to_col = tile_to

        from_x = from_col * (tile_width_px + tile_padding_width_px) + tile_padding_width_px + tile_width_px // 2
        from_y = from_row * (tile_width_px + tile_padding_width_px) + tile_padding_width_px + tile_width_px // 2
        to_x = to_col * (tile_width_px + tile_padding_width_px) + tile_padding_width_px + tile_width_px // 2
        to_y = to_row * (tile_width_px + tile_padding_width_px) + tile_padding_width_px + tile_width_px // 2

        pygame.draw.line(surface, grid_line_color, (pos_offset[0] + from_x, pos_offset[1] + from_y), (pos_offset[0] + to_x, pos_offset[1] + to_y), 3)

    return None

if __name__ == "__main__":
    num_rows = 7
    num_cols = 5

    total_width = num_cols * tile_width_px + (num_cols + 1) * tile_padding_width_px
    total_height = num_rows * tile_width_px + (num_rows + 1) * tile_padding_width_px

    pygame.init()
    
    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Grid Drawing Example")

    clock = pygame.time.Clock()

    board = np.zeros((num_rows, num_cols), dtype=int)

    # Example board setup
    # // // // // //
    # // // // // //
    # // // // // //
    # // // // // //
    #  4 16 //  2 1024
    # 512 64 64 64 256
    # 16  8 64 32 128

    board[6, 0] = 16
    board[6, 1] = 8
    board[6, 2] = 64
    board[6, 3] = 32
    board[6, 4] = 128
    board[5, 0] = 512
    board[5, 1] = 64
    board[5, 2] = 64
    board[5, 3] = 64
    board[5, 4] = 256
    board[4, 0] = 4
    board[4, 1] = 16
    board[4, 3] = 2
    board[4, 4] = 1024

    # Example merge indicators: 64's merge in to (5, 2)

    selected_col = 2
    selected_col_value = 64
    merge_indicators = [
        ((6, 2), (5, 2)),
        ((5, 1), (5, 2)),
        ((5, 3), (5, 2))
    ]

    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        screen.fill(background_color)
        draw_grid(screen, (0, 0), board, selected_col, selected_col_value, merge_indicators)

        pygame.display.flip()
        clock.tick(60)
