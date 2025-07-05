import sys
import pygame

tile_width_px = 92
tile_font_size_px = 36
tile_font_color = (255, 255, 255)
background_color = (41, 45, 53)

TILE_COLORS = {
    2:    (222, 98,  196),
    4:    (133, 184,  75),
    8:    ( 69, 202, 201),
    16:   ( 98, 129, 225),
    32:   (228, 137,  87),
    64:   (138, 125, 255),
    128:  (141, 184, 202),
    256:  (252,  82, 116),
    512:  (  2, 189, 113),
    1024: (133, 151, 150),
    2048: (133, 151, 150),
    4096: (133, 151, 150),
    8192: (133, 151, 150),
    16384: (133, 151, 150),
    32768: (133, 151, 150),
    65536: (133, 151, 150),
}

TILE_SURFACES = {}

def draw_tile_precompute_surface(value: int, downscale: float = 1, aa_scale: int = 1):
    key = f"{value}_{downscale}_{aa_scale}"
    if key in TILE_SURFACES:
        return TILE_SURFACES[key]
    
    if value not in TILE_COLORS:
        raise ValueError(f"Unsupported tile value: {value}")
    
    surface = pygame.Surface((tile_width_px * aa_scale, tile_width_px * aa_scale))
    surface.fill(background_color)

    rect = pygame.Rect(0, 0, tile_width_px * aa_scale, tile_width_px * aa_scale)
    pygame.draw.rect(
        surface,
        TILE_COLORS[value],
        rect,
        border_radius=4 * aa_scale
    )

    text = str(value)
    font = pygame.font.Font("Roboto-Bold.ttf", tile_font_size_px * aa_scale)
    text_surface = font.render(text, True, tile_font_color)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)

    if aa_scale > 1:
        surface = pygame.transform.smoothscale(surface, (int(tile_width_px / downscale), int(tile_width_px / downscale)))

    TILE_SURFACES[key] = surface
    return surface

def draw_tile(surface: pygame.Surface, value: int, pos_offset: tuple[int, int] = (0, 0), downscale: float = 1, aa_scale: int = 1):
    surface_for_tile = draw_tile_precompute_surface(value, downscale, aa_scale)
    rect = surface_for_tile.get_rect(topleft=pos_offset)
    surface.blit(surface_for_tile, rect)
    return


if __name__ == "__main__":
    # Draw a row of tiles, with a margin of 15 pixels around the edges and between them

    pygame.init()
    num_tiles_to_draw = len(TILE_COLORS)
    margin = 15
    total_width = (num_tiles_to_draw + 1) * margin + num_tiles_to_draw * tile_width_px
    total_height = tile_width_px + 2 * margin

    screen = pygame.display.set_mode((total_width, total_height))

    pygame.display.set_caption("Tile Drawing Example")

    clock = pygame.time.Clock()

    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(background_color)
        for i, (value, color) in enumerate(TILE_COLORS.items()):
            pos_offset = (margin + i * (tile_width_px + margin), margin)
            draw_tile(screen, value, pos_offset, aa_scale=4)
        
        pygame.display.flip()
        clock.tick(60)
