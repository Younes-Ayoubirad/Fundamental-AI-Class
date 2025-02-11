import random

import pygame
from game import AngryGame, PygameInit
from minimax_agent import MinimaxAgent
from queue import Queue
import copy



if __name__ == "__main__":
    env = AngryGame(template='test')
    agent = MinimaxAgent(env, max_depth=6)
    screen, clock = PygameInit.initialization()
    FPS = 20
    env.reset()
    counter = 0

    running = True
    while running:
        if AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions):
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if counter % 2 == 0:
            action = agent.get_best_action(env.grid)
            agent.num_action+=1
            env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 2 == 1:
            env.queen_step()
            env.render(screen)
            if AngryGame.is_lose(env.grid, env.num_actions):
                running = False

        counter += 1
        pygame.display.flip()
        clock.tick(FPS)
        print(f'Current Score == {AngryGame.calculate_score(env.grid, env.num_actions)}')

    pygame.quit()

