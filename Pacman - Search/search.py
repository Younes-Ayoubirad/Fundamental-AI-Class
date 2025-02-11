# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):

    search_stack = util.Stack()
    current_state = problem.getStartState()
    search_stack.push((current_state, []))
    visited_stats = set()

    while not search_stack.isEmpty():
        state, path = search_stack.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited_stats:
            visited_stats.add(state)
            for successor_state, successor_action, successor_stepCost in problem.getSuccessors(state):
                successor_path = path + [successor_action]
                search_stack.push((successor_state, successor_path))


def breadthFirstSearch(problem):
    from util import Queue

    frontier = Queue()
    start_state = problem.getStartState()
    frontier.push((start_state, []))
    visited_states = set()

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        if current_state in visited_states:
            continue

        visited_states.add(current_state)

        if problem.isGoalState(current_state):
            return path

        for successor_state, action, step_cost in problem.getSuccessors(current_state):
            if successor_state not in visited_states:
                frontier.push((successor_state, path + [action]))
    return []


def RecFunc(problem, current):
    if problem.isGoalState(current):
        return current
    else:
        for successor in current.getSuccessors(current):
            result = RecFunc(problem, successor)
            if result: return True
        return False



def uniformCostSearch(problem):
    Pqueue = util.PriorityQueue()
    start_state = problem.getStartState()

    visited_states = {}
    Pqueue.push((start_state, 0, []), 0)

    while not Pqueue.isEmpty():
        current_state, current_cost, path = Pqueue.pop()

        if current_state in visited_states and visited_states[current_state] <= current_cost:
            continue

        visited_states[current_state] = current_cost

        if problem.isGoalState(current_state):
            return path

        for successor_state, successor_action, successor_stepCost in problem.getSuccessors(current_state):
            successor_cost = current_cost + successor_stepCost
            successor_path = path + [successor_action]
            if successor_state not in visited_states or successor_cost < visited_states[successor_state]:
                Pqueue.push((successor_state, successor_cost, successor_path), successor_cost)
    return []



def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic):
    from util import PriorityQueue

    Pqueue = PriorityQueue()

    start_state = problem.getStartState()

    visited_states = {}

    Pqueue.push((start_state, 0, []), 0 + heuristic(start_state, problem))

    while not Pqueue.isEmpty():
        current_state, current_cost, path = Pqueue.pop()

        if current_state in visited_states and visited_states[current_state] <= current_cost:
            continue

        visited_states[current_state] = current_cost

        if problem.isGoalState(current_state):
            return path

        for successor_state, successor_action, successor_stepCost in problem.getSuccessors(current_state):
            successor_cost = current_cost + successor_stepCost
            total_cost = successor_cost + heuristic(successor_state, problem)

            if successor_state not in visited_states or successor_cost < visited_states[successor_state]:
                successor_path = path + [successor_action]
                Pqueue.push((successor_state, successor_cost, successor_path), total_cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
