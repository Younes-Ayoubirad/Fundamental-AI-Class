import math

from game import AngryGame
from queue import Queue
import copy


class MinimaxAgent:
    def __init__(self, env, max_depth):
        self.env = env
        self.max_depth = max_depth
        self.max_goal_distance = 50

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
            index, depth = que.get()
            for i in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if not (needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]):
                    if cloned[needed_index[0]][needed_index[1]] in ['T']:
                        cloned[needed_index[0]][needed_index[1]] = 'R'
                        que.put((needed_index, depth + 1))
                    if cloned[needed_index[0]][needed_index[1]] == 'E':
                        return depth
        return 0

    def find_closet_path_shooter(self, grid):
        cloned = copy.deepcopy(grid)
        hen_pos = AngryGame.get_hen_position(cloned)
        que = Queue()
        que.put((hen_pos, 1))
        while not que.empty():
            index, depth = que.get()
            for i in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]:
                    pass
                elif cloned[needed_index[0]][needed_index[1]] == 'T':
                    cloned[needed_index[0]][needed_index[1]] = 'R'
                    que.put((needed_index, depth + 1))
                elif cloned[needed_index[0]][needed_index[1]] == 'S':
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
                elif cloned[needed_index[0]][needed_index[1]] == 'T':
                    cloned[needed_index[0]][needed_index[1]] = 'R'
                    que.put((needed_index, depth + 1,))
                elif cloned[needed_index[0]][needed_index[1]] == 'H':
                    return depth

    def find_closet_path_hen(self, grid):
        cloned = copy.deepcopy(grid)
        hen_pos = AngryGame.get_queen_position(cloned)
        shooter = AngryGame.get_slingshot_position(cloned)
        if shooter is None:
            return 0
        cloned[shooter[0]][shooter[1]] = 'R'
        que = Queue()
        que.put((hen_pos, 1))
        while not que.empty():
            index, depth = que.get()
            cloned[index[0]][index[1]] = 'R'
            for i in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                needed_index = [index[0] + i[0], index[1] + i[1]]
                if needed_index[0] in [-1, 10] or needed_index[1] in [-1, 10]:
                    pass
                elif cloned[needed_index[0]][needed_index[1]] == 'T':
                    que.put((needed_index, depth + 1))
                elif cloned[needed_index[0]][needed_index[1]] == 'H':
                    return depth

    def minimax(self, grid, depth, is_maximizing, alpha, beta):
        if depth == 0 or AngryGame.is_win(grid) or AngryGame.is_lose(grid, 150):
            return self.evaluate(grid)

        if is_maximizing:
            max_eval = -float('inf')
            for successor, action in AngryGame.generate_hen_successors(grid):
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

        best_action = None
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for successor, action in AngryGame.generate_hen_successors(grid):
            eval_value = self.minimax(successor, self.max_depth - 1, False, alpha, beta)
            if eval_value > max_eval:
                max_eval = eval_value
                best_action = action

        return best_action

    def evaluate(self, grid):
        hen_queen = self.find_closet_path_hen(grid)
        goal_distance = self.find_closet_path_egg(grid)
        eggs = AngryGame.get_egg_coordinate(grid)

        pigs = AngryGame.get_pig_coordinate(grid)
        if goal_distance is None:
            goal_distance = 0
        goal_eval = (self.max_goal_distance- goal_distance)/self.max_goal_distance
        if len(eggs) == 0:
            goal = self.find_closet_path_shooter(grid)
            if goal is None:
                goal = -100
            goal_eval = (self.max_goal_distance-goal)/self.max_goal_distance*800
        queen_function = math.pow(((self.max_goal_distance - hen_queen) / self.max_goal_distance), 2) * 1000
        return goal_eval - queen_function + (8 - len(eggs)) * 250 - ((8 - len(pigs)) * 250)
