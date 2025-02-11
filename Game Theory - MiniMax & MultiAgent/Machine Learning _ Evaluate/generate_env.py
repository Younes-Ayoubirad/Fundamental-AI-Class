import random

def generate_random_grid(grid_size, num_pigs, num_queens, num_rocks, num_eggs, slingshot=True):
    grid = [['T' for _ in range(grid_size)] for _ in range(grid_size)]
    filled_spaces = []

    if slingshot:
        grid[grid_size - 1][grid_size - 1] = 'S'
        filled_spaces.append((grid_size - 1, grid_size - 1))

    while True:
        r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        if (r, c) not in filled_spaces:
            grid[r][c] = 'H'
            filled_spaces.append((r, c))
            break

    for _ in range(num_pigs):
        while True:
            r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (r, c) not in filled_spaces:
                grid[r][c] = 'P'
                filled_spaces.append((r, c))
                break

    for _ in range(num_queens):
        while True:
            r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (r, c) not in filled_spaces:
                grid[r][c] = 'Q'
                filled_spaces.append((r, c))
                break

    for _ in range(num_rocks):
        while True:
            r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (r, c) not in filled_spaces:
                grid[r][c] = 'R'
                filled_spaces.append((r, c))
                break

    for _ in range(num_eggs):
        while True:
            r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (r, c) not in filled_spaces:
                grid[r][c] = 'E'
                filled_spaces.append((r, c))
                break

    return grid


def save_grid_to_txt(grid, file_name):

    with open(file_name, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')


if __name__ == "__main__":

    num_envs = 10
    grid_size = 10
    num_pigs = 8
    num_queens = 1
    num_rocks = 8
    num_eggs = 8

    for i in range(num_envs):
        grid = generate_random_grid(grid_size, num_pigs, num_queens, num_rocks, num_eggs)

        file_name = f"random_env_{i + 1}.txt"
        save_grid_to_txt(grid, file_name)

