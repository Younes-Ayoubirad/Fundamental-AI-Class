import math
import queue
import random

import numpy as np

from game import AngryGame
from queue import Queue
import copy

EGG_REWARD = 0.2
PIG_REWARD = -0.2
DEFAULT_REWARD = (-0.1)
LOSE_REWARD = -0.5
SLING_REWARD = 1

EGGS = 8
PIGS = 8
QUEEN = 1

MAX_ACTIONS = 150


class Red:
    def __init__(self, env, max_depth):
        self.is_alive = True
        self.env = env
        self.max_depth = max_depth
        self.max_distance = 50
        self.max_goal_distance = 50
        self.reward = 0
    def set_reward(self, reward):
        self.reward = reward
    def find_closet_path_hen(self, grid):
        cloned = copy.deepcopy(grid)
        queen_state = AngryGame.get_queen_position(cloned)
        shooter = AngryGame.get_slingshot_position(cloned)
        if shooter is None:
            return 0, []
        cloned[shooter[0]][shooter[1]] = 'R'
        que = Queue()
        que.put((queen_state, 1, [[queen_state[0], queen_state[1]]]))
        while not que.empty():
            index, depth, path = que.get()
            for i in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]:
                    continue
                elif cloned[needed_index[0]][needed_index[1]] == 'T':
                    cloned[index[0]][index[1]] = 'R'
                    que.put((needed_index, depth + 1, path + [needed_index]))
                elif cloned[needed_index[0]][needed_index[1]] == 'H':
                    return depth, path + [needed_index]
        return 0,[]
    def find_closet_path_egg(self, grid):
        cloned = copy.deepcopy(grid)
        hen_pos = AngryGame.get_hen_position(grid)
        shooter = AngryGame.get_slingshot_position(cloned)
        if shooter is None:
            return 0
        cloned[shooter[0]][shooter[1]] = 'R'
        que = Queue()
        que.put((hen_pos, 1))
        while not que.empty():
            index, depth= que.get()
            for i in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if not( needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]):
                    if cloned[needed_index[0]][needed_index[1]] in ['T', 'Q', 'B']:
                        cloned[needed_index[0]][needed_index[1]] = 'R'
                        que.put((needed_index, depth + 1))
                    if cloned[needed_index[0]][needed_index[1]] == 'E':
                        return depth
        return 0
    def find_closet_path_queen(self, grid):
        cloned = copy.deepcopy(grid)
        red_position = AngryGame.get_bird_position(cloned)
        que = Queue()
        que.put((red_position, 0))
        while not que.empty():
            index, depth = que.get()
            for i in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]:
                    pass
                elif cloned[needed_index[0]][needed_index[1]] in ['T','B']:
                    cloned[needed_index[0]][needed_index[1]] = 'R'
                    que.put((needed_index, depth + 1,))
                elif cloned[needed_index[0]][needed_index[1]] == 'Q':
                    return depth
    def find_closet_path_hen_bird(self, grid):
        cloned = copy.deepcopy(grid)
        red_position = AngryGame.get_bird_position(cloned)
        shooter = AngryGame.get_slingshot_position(cloned)
        if shooter is None:
            return 100
        cloned[shooter[0]][shooter[1]] = 'R'
        que = Queue()
        que.put((red_position, 1))
        while not que.empty():
            index, depth = que.get()
            for i in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]:
                    pass
                elif cloned[needed_index[0]][needed_index[1]] in ['T','Q']:
                    cloned[needed_index[0]][needed_index[1]] = 'R'
                    que.put((needed_index, depth + 1,))
                elif cloned[needed_index[0]][needed_index[1]] == 'H':
                    return depth

    def minimax(self, grid, depth, is_maximizing, alpha, beta):
        if depth == 0 or AngryGame.is_win(grid) or AngryGame.is_lose(grid, 150):
            return self.evaluate(grid)

        if is_maximizing:
            max_eval = -float('inf')
            for successor, action in AngryGame.generate_bird_successors(grid):
                eval_value = self.minimax(successor, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for successor, action in AngryGame.generate_queen_successors(grid):
                eval_value = self.minimax(successor, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval
    def get_best_action(self, grid):
        if not self.env.is_queen_exists(grid):
            return random.choice([0, 1, 2, 3])
        best_action = None
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        list = AngryGame.generate_bird_successors(grid)
        list.reverse()
        for next_successor, next_actions in list:
            eval_value = self.minimax(next_successor, self.max_depth - 1, False, alpha, beta)
            if eval_value > max_eval:
                max_eval = eval_value
                best_action = next_actions
        return best_action
    def calculate(self,number):
        return (-number+self.max_distance)/self.max_distance


    def evaluate(self, grid):
        eggs = AngryGame.get_egg_coordinate(grid)
        if AngryGame.is_win(grid):
            if self.reward > 1400 or len(eggs) < 3:
                return 1000
            else:
                return -1000
        queen_bird = self.find_closet_path_queen(grid)
        queen_hen,path = self.find_closet_path_hen(grid)
        return  self.calculate(queen_bird)- self.calculate(queen_hen)

