import numpy as np
import pygame

import Red_agent
import Hen_agent
from game import AngryGame, PygameInit
from queue import Queue
import copy


if __name__ == "__main__":

    env = AngryGame(template='hard')
    terence = Red_agent.Red(env, max_depth=6)
    mathilda = Hen_agent.MinimaxAgent(env, max_depth=6)
    screen, clock = PygameInit.initialization()
    FPS = 10

    env.reset()
    counter = 0
    running = True
    while running:
        if AngryGame.is_win(env.grid) or AngryGame.is_lose(env.grid, env.num_actions):
           running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if counter % 3 == 0:
            action = mathilda.get_best_action(env.grid)
            env.hen_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 1:
            terence.set_reward(AngryGame.calculate_score(env.grid, env.num_actions))
            action = terence.get_best_action(env.grid)
            env.bird_step(action)
            env.render(screen)
            if AngryGame.is_win(env.grid):
                running = False

        if counter % 3 == 2:
            env.queen_step()
            env.render(screen)
            if AngryGame.is_lose(env.grid, env.num_actions):
                running = False

        counter += 1
        pygame.display.flip()
        clock.tick(FPS)
        print(f'Current Score == {AngryGame.calculate_score(env.grid, env.num_actions)}')

    pygame.quit()

