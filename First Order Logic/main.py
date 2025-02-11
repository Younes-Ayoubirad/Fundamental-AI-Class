from environment import FirstOrderAngry, PygameInit
import pygame
from pyswip import Prolog

prolog = Prolog()
steps = []

def initialize_prolog(prolog, agent, obstacle):
    prolog.consult("knowledgebase.pl")

    prolog.retractall("grid_size(_, _)")
    prolog.retractall("agent(_, _)")
    prolog.retractall("obstacle(_, _)")

    prolog.assertz(f"grid_size({8}, {8})")

    prolog.assertz(f"agent({agent[0]}, {agent[1]})")

    for obstacle in obstacle:
        prolog.assertz(f"obstacle({obstacle[0]}, {obstacle[1]})")


def extract_environment_data(grid):
    agent = None
    obstacle = []

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 'B':
                agent = (r + 1, c + 1)
            elif cell == 'R':
                obstacle.append((r + 1, c + 1))

    return agent, obstacle


def get_pig(arr):
    p_indexes = []
    for i in range(8):
        for j in range(8):
            if arr[i][j] == 'P':
                p_indexes.append((i, j))
    return p_indexes


def find_path(start, goal):
    depth = [64, 128, 256, 512, 1024, 2048, 4096, 10000, 15000, 20000, 50000]
    for i in depth:
        query = f"shortest_path(({start[0]}, {start[1]}), ({goal[0]}, {goal[1]}), Path,{i})"
        result = list(prolog.query(query))
        if result:
             return result[0]["Path"]


def choose_path(env):
    bird = env.get_bird_position()

    pigs = get_pig(env.grid)
    nbird = [bird[0] + 1, bird[1] + 1]
    shortest_path = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for pig in pigs:
        next = [pig[0] + 1, pig[1] + 1]
        path = find_path(nbird, next)
        if path is not None and len(shortest_path) > len(path):
            shortest_path = path
    return shortest_path


def get_action(bird, next):
    state = [bird[0] + 1, bird[1] + 1]
    distance = [int(next[2]) - state[0], int(next[5]) - state[1]]
    if distance[0] == -1:
        return 0
    if distance[0] == 1:
        return 1
    if distance[1] == -1:
        return 2
    return 3


if __name__ == "__main__":
    env = FirstOrderAngry(template='simple')

    screen, clock = PygameInit.initialization()
    FPS = 8
    env.reset()

    agent, obstacle = extract_environment_data(env.grid)
    initialize_prolog(prolog, agent, obstacle)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if len(steps) == 0:
            steps = list(choose_path(env))
            steps.pop(0)
        print(steps)
        action = get_action(env.get_bird_position(), steps.pop(0))
        bird_pos, is_win = env.bird_step(action)
        env.render(screen)
        if is_win:
            print(f'Win')
            running = False
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
