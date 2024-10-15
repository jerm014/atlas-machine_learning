import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GRID_SIZE = 8
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
ANIMATION_SPEED = 5

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Match-3 Game")

# Create the grid
grid = [[random.choice(COLORS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            pygame.draw.rect(screen, grid[y][x], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, (0, 0, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def animate_swap(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    distance = CELL_SIZE
    steps = CELL_SIZE // ANIMATION_SPEED

    for step in range(steps + 1):
        screen.fill((255, 255, 255))
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if (x, y) == pos1:
                    dx = (x2 - x1) * step * ANIMATION_SPEED
                    dy = (y2 - y1) * step * ANIMATION_SPEED
                    pygame.draw.rect(screen, grid[y1][x1], (x * CELL_SIZE + dx, y * CELL_SIZE + dy, CELL_SIZE, CELL_SIZE))
                elif (x, y) == pos2:
                    dx = (x1 - x2) * step * ANIMATION_SPEED
                    dy = (y1 - y2) * step * ANIMATION_SPEED
                    pygame.draw.rect(screen, grid[y2][x2], (x * CELL_SIZE + dx, y * CELL_SIZE + dy, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(screen, grid[y][x], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, (0, 0, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        pygame.display.flip()
        pygame.time.wait(30)

def swap_cells(pos1, pos2):
    animate_swap(pos1, pos2)
    x1, y1 = pos1
    x2, y2 = pos2
    grid[y1][x1], grid[y2][x2] = grid[y2][x2], grid[y1][x1]

def check_matches():
    matches = set()
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            # Check 2x2 grid
            if x < GRID_SIZE - 1 and y < GRID_SIZE - 1:
                if grid[y][x] == grid[y][x+1] == grid[y+1][x] == grid[y+1][x+1]:
                    matches.update([(x, y), (x+1, y), (x, y+1), (x+1, y+1)])
    return matches

def remove_matches(matches):
    for x, y in matches:
        grid[y][x] = None

def fall_blocks():
    for x in range(GRID_SIZE):
        empty_cells = []
        for y in range(GRID_SIZE - 1, -1, -1):
            if grid[y][x] is None:
                empty_cells.append(y)
            elif empty_cells:
                new_y = empty_cells.pop(0)
                grid[new_y][x] = grid[y][x]
                grid[y][x] = None
                empty_cells.append(y)
        
        for y in empty_cells:
            grid[y][x] = random.choice(COLORS)

def animate_fall():
    falling = True
    while falling:
        falling = False
        screen.fill((255, 255, 255))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE - 1, 0, -1):
                if grid[y][x] is None and grid[y-1][x] is not None:
                    grid[y][x], grid[y-1][x] = grid[y-1][x], grid[y][x]
                    falling = True
        draw_grid()
        pygame.display.flip()
        pygame.time.wait(50)
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[y][x] is None:
                grid[y][x] = random.choice(COLORS)
    draw_grid()
    pygame.display.flip()

def main():
    selected = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked = (pos[0] // CELL_SIZE, pos[1] // CELL_SIZE)
                if selected:
                    if (abs(clicked[0] - selected[0]) == 1 and clicked[1] == selected[1]) or \
                       (abs(clicked[1] - selected[1]) == 1 and clicked[0] == selected[0]):
                        swap_cells(selected, clicked)
                        matches = check_matches()
                        if matches:
                            remove_matches(matches)
                            animate_fall()
                        selected = None
                    else:
                        selected = clicked
                else:
                    selected = clicked

        screen.fill((255, 255, 255))
        draw_grid()
        if selected:
            pygame.draw.rect(screen, (255, 255, 255), 
                             (selected[0] * CELL_SIZE, selected[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
