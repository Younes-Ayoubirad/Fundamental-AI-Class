import pygame
import numpy as np
import copy

GRID_SIZE = 8
TILE_SIZE = 100
background = (106, 70, 205)

class PygameInit:
    @classmethod
    def initialization(cls):
        grid_size_x = GRID_SIZE
        grid_size_y = GRID_SIZE
        tile_size = TILE_SIZE

        pygame.init()
        screen = pygame.display.set_mode((grid_size_x * tile_size, grid_size_y * tile_size))
        pygame.display.set_caption("FOL")
        clock = pygame.time.Clock()

        return screen, clock

class FirstOrderAngry:
    def __init__(self, template: str):
        self.__grid_size = GRID_SIZE
        self.__tile_size = TILE_SIZE
        self.__template_name = template

        self.__base_grid = self.__generate_grid()
        self.grid = copy.deepcopy(self.__base_grid)
        self.__base_grid = copy.deepcopy(self.grid)

        self.__bird_image = pygame.image.load('Env/icons/angry-birds.png')
        self.__bird_image = pygame.transform.scale(self.__bird_image, (self.__tile_size, self.__tile_size))
        self.__bird_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__bird_with_background.fill(background)
        self.__bird_with_background.blit(self.__bird_image, (0, 0))

        self.__pig_image = pygame.image.load('Env/icons/pigs.png')
        self.__pig_image = pygame.transform.scale(self.__pig_image, (self.__tile_size, self.__tile_size))
        self.__pig_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__pig_with_background.fill(background)
        self.__pig_with_background.blit(self.__pig_image, (0, 0))

        self.__rock_image = pygame.image.load('Env/icons/rocks.png')
        self.__rock_image = pygame.transform.scale(self.__rock_image, (self.__tile_size, self.__tile_size))
        self.__rock_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__rock_with_background.fill(background)
        self.__rock_with_background.blit(self.__rock_image, (0, 0))


    def __generate_grid(self):
        grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        with open(f'Env/templates/{self.__template_name}.txt') as file:
            template_str = file.readlines()

        for i in range(self.__grid_size):
            for j in range(self.__grid_size):
                grid[i][j] = template_str[i][j]

        return grid

    def reset(self):
        self.grid = copy.deepcopy(self.__base_grid)

    def bird_step(self, action):
        bird_pos = self.get_bird_position()
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        dx, dy = actions[action]
        new_row, new_col = bird_pos[0] + dx, bird_pos[1] + dy

        if self.__is_valid_for_bird_position(self.grid, new_row, new_col):
            self.grid[bird_pos[0]][bird_pos[1]] = 'T'
            bird_pos = (new_row, new_col)
            self.grid[bird_pos[0]][bird_pos[1]] = 'B'

        is_win = self.is_win()
        return bird_pos, is_win


    def get_bird_position(self):
        for r in range(len(self.grid)):
            for c in range(len(self.grid)):
                if self.grid[r][c] == 'B':
                    return tuple([r, c])


    def is_win(self):
        for r in range(len(self.grid)):
            for c in range(len(self.grid)):
                if self.grid[r][c] == 'P':
                    return False
        return True


    def render(self, screen):
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = background
                pygame.draw.rect(screen, color, (c * self.__tile_size, r * self.__tile_size, self.__tile_size,
                                                 self.__tile_size))
                if self.grid[r][c] == 'B':
                    screen.blit(self.__bird_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'R':
                    screen.blit(self.__rock_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'P':
                    screen.blit(self.__pig_with_background, (c * self.__tile_size, r * self.__tile_size))

        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size), (self.__grid_size * self.__tile_size,
                                                                            r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0), (c * self.__tile_size,
                                                                            self.__grid_size * self.__tile_size), 2)

    @classmethod
    def __is_valid_for_bird_position(cls, grid, new_row, new_col):
        return (0 <= new_row < len(grid)
                and 0 <= new_col < len(grid)
                and grid[new_row][new_col] != 'R')