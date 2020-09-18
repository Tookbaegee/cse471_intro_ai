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

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.c
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from game import Directions
    from game import Actions

    stateStack = util.Stack()
    actions = []
    visited = set()
    startState = problem.getStartState()
    stateStack.push((startState, []))
    while not stateStack.isEmpty():
        currentState, currentAction = stateStack.pop()
       
        if problem.isGoalState(currentState):
            actions = currentAction
            break
        if not currentState in visited:
            visited.add(currentState)
            for triple in reversed(problem.getSuccessors(currentState)):
                if triple[0] not in visited:
                    nextAction = currentAction + [triple[1]]
                    stateStack.push((triple[0], nextAction))
    
    return actions
 

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from game import Directions
    from game import Actions
    from searchAgents import CornersProblem
    
    stateQueue = util.Queue()
    actions = []
    visited = set()
    currentState = problem.getStartState()
    stateQueue.push((currentState, []))

    while not stateQueue.isEmpty():
        currentState, currentAction = stateQueue.pop()

        if problem.isGoalState(currentState):
            actions += currentAction
            break
        if currentState not in visited:
            visited.add(currentState)
            for triple in problem.getSuccessors(currentState):
                if triple[0] not in visited:
                    nextAction = currentAction + [triple[1]]
                    stateQueue.push((triple[0], nextAction))
                
    return actions


def iterativeDeepeningSearch(problem):
    limit = 0
    while 1:

        stateStack = util.Stack()
        actions = []
        visited = set()
        startState = problem.getStartState()
        #stateNode = (state, path, depth)
        stateStack.push((startState, [], 0))
        while not stateStack.isEmpty():

            currentState, currentAction, currentDepth = stateStack.pop()
   
            if problem.isGoalState(currentState):
                actions = currentAction
                return actions

            successors = []
            if currentState not in visited:
                if currentDepth < limit:
                    successors = problem.getSuccessors(currentState)

                else:
                    problem.getSuccessors(currentState)

                visited.add(currentState)

                for triple in reversed(successors):
                    if triple[0] not in visited and triple [0] not in [state[0] for state in stateStack.list]:
                        nextAction = currentAction + [triple[1]]
                        stateStack.push((triple[0], nextAction, currentDepth + 1))
            
        limit += 1

    return actions
        
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startState = (problem.getStartState(), [], 0)
    frontier = util.PriorityQueue()
    frontier.push(startState, 0)
    explored = set()
    
    while not frontier.isEmpty():
    
        currentState, currentAction, currentCost = frontier.pop()
        if problem.isGoalState(currentState):
            return currentAction
        if not currentState in explored:
            explored.add(currentState)
            for child in problem.getSuccessors(currentState):
                pathCost = child[2] + currentCost
                frontier.push((child[0], currentAction + [child[1]], pathCost), pathCost)
                

    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = (problem.getStartState(), [], 0)
    frontier = util.PriorityQueue()
    frontier.push(startState, 0)
    explored = set()
    
    while not frontier.isEmpty():
    
        currentState, currentAction, currentCost = frontier.pop()
        if problem.isGoalState(currentState):
            return currentAction
        if not currentState in explored:
            explored.add(currentState)
            for child in problem.getSuccessors(currentState):
                pathCost = child[2] + currentCost
                heuristicCost = pathCost + heuristic(child[0], problem)
                frontier.push((child[0], currentAction + [child[1]], pathCost), heuristicCost)
                


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
iddfs = iterativeDeepeningSearch