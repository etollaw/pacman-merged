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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

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
          state: Search state

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# -----------------------------
# Q1: Depth-First Search (DFS)
# Implements graph search using a stack (LIFO). Explores as deep as possible before backtracking.
# Returns a list of actions to reach the goal.
def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.
    Returns a list of actions that reaches the goal using DFS (graph search).
    """
    print("start state: ", problem.getStartState())
    print("goal state: ", problem.isGoalState(problem.getStartState()))
    print("successors: ", problem.getSuccessors(problem.getStartState()))
    fringe = util.Stack()
    visited = set()
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        state, path = fringe.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    fringe.push((successor, path + [action]))
    return []  # No solution found

"*** YOUR CODE HERE ***"
    
    # fringe = util.Stack()
    # seen = set()
    # fringe.push((problem.getStartState(), []))
    
    # while not fringe.isEmpty():
        
    #     state, path = fringe.pop() 
        
    #     if problem.isGoalState(state):
    #         return path
        
    #     if state not in seen:
    #         seen.add(state)
            
    #         for successor, action, stepCost in problem.getSuccessors(state):
    #             if successor not in seen:
    #                 fringe.push((successor, path + [action]))   
    # return []  # Goal not found
    
    # util.raiseNotDefined()


# -----------------------------
# Q2: Breadth-First Search (BFS)
# Implements graph search using a queue (FIFO). Explores all nodes at the current depth before going deeper.
# Returns a list of actions to reach the goal.
def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the shallowest nodes in the search tree first.
    Returns a list of actions that reaches the goal using BFS (graph search).
    """
    fringe = util.Queue()
    visited = set()
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        state, path = fringe.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    fringe.push((successor, path + [action]))
    return []  # No solution found

# -----------------------------
# Q3: Uniform Cost Search (UCS)
# Expands the node with the lowest path cost. Uses a priority queue.
# Returns a list of actions to reach the goal, optimal for variable step costs.
def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the node of least total cost first (UCS).
    Returns a list of actions that reaches the goal using UCS (graph search).
    """
    fringe = util.PriorityQueue()
    visited = set()
    # (state, path, cost)
    fringe.push((problem.getStartState(), [], 0), 0)
    best_cost = dict()

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited or cost < best_cost.get(state, float('inf')):
            visited.add(state)
            best_cost[state] = cost
            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                if successor not in visited or new_cost < best_cost.get(successor, float('inf')):
                    fringe.push((successor, path + [action], new_cost), new_cost)
    return []  # No solution found

# -----------------------------
# Q4: Null Heuristic
# Returns 0 for all states. Used as a default for A* to make it behave like UCS.
def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# -----------------------------
# Q4: A* Search
# Expands the node with the lowest (cost + heuristic). Uses a priority queue.
# Returns a list of actions to reach the goal, optimal if heuristic is admissible.
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """
    Search the node that has the lowest combined cost and heuristic first (A*).
    Returns a list of actions that reaches the goal using A* (graph search).
    """
    fringe = util.PriorityQueue()
    visited = set()
    # (state, path, cost)
    start = problem.getStartState()
    fringe.push((start, [], 0), heuristic(start, problem))
    best_cost = dict()

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited or cost < best_cost.get(state, float('inf')):
            visited.add(state)
            best_cost[state] = cost
            for successor, action, stepCost in problem.getSuccessors(state):
                new_cost = cost + stepCost
                h = heuristic(successor, problem)
                if successor not in visited or new_cost < best_cost.get(successor, float('inf')):
                    fringe.push((successor, path + [action], new_cost), new_cost + h)
    return []  # No solution found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
