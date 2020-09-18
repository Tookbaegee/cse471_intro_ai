# multiAgents.py
# --------------
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

import json
import csp
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """
        print("new Pos: {}".format(newPos))
        print("new Food: {}".format(newFood))
        print("new Ghost States: {}".format(newGhostStates))
        print("new Scared Times {}".format(newScaredTimes))
        """
        #weight of food and ghost  to propagate most favorable reward to decide next action
        #food more favored to challenge the ghosts. (works well for one ghost)
        food = 6.0
        ghost = 1.0

        currentScore = successorGameState.getScore()

        #negative reward based on weight/distance from ghost
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostDistance > 0 :
            currentScore -= ghost / ghostDistance
        
        #positive reward based on weight/distance from closest food.
        foodDistance = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(foodDistance) > 0:
            currentScore += food / min(foodDistance)

        return currentScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

#used to check if node is leaf.
def isLeaf(self, state, depth, agent):
            return depth == self.depth or state.isWin() or state.isLose() or len(state.getLegalActions(agent)) == 0
#check if the agent is maximizing agent.
def isMaximizer(state, agent):
    return agent % state.getNumAgents() == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        

        def minimax(state, depth, agent):
            #if agent is the maximizing agent (at 0) continue recursing down minimax as agent 0
            if agent == state.getNumAgents():  
                return minimax(state, depth + 1, 0)  

            # if leaf, return value of the node
            if isLeaf(self, state, depth, agent):
                return self.evaluationFunction(state)  
            
            successors = set()
            #expand adversary based on current agent's successors
            for action in state.getLegalActions(agent):
                successors.add(minimax(state.generateSuccessor(agent, action), depth, agent + 1))
                
            #if agent is maximizing, return the max of suceesors, else, min based on scores from minimax call
            if isMaximizer(state, agent):
                return max(successors)
            else:
                return min(successors)

        #among the possible legal action, choose the max based on the minimax score of current state's successors.
        return max(gameState.getLegalActions(0), key = lambda x: minimax(gameState.generateSuccessor(0, x), 0, 1))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def getAlphaBeta(state, depth, agent, alpha, beta):
            
            minmax = None
            bestValue = 0
            bestAction = None
            #if Maximizer, set for change in alpha value, else for beta.
            if isMaximizer(state, agent):
                minmax = max
                bestValue = float('-inf')
            else:
                minmax = min
                bestValue = float('inf')
            #expand to all possible children and recursively prune the next agent based on the children to get value
            for action in state.getLegalActions(agent):

                successor = state.generateSuccessor(agent, action)
                value,_ = prune(successor, depth, agent + 1, alpha, beta)
                bestValue, bestAction = minmax((value, action), (bestValue, bestAction))
                #if maximizer update alpha value if best value is greater than current beta
                if isMaximizer(state, agent):
                    if bestValue > beta:
                        return bestValue, bestAction
                    alpha = minmax(alpha, bestValue)
               
                #if minimizer update beta value if best value is less than current alpha
                else:
                    if bestValue < alpha:
                        return bestValue, bestAction
                    beta = minmax(beta, bestValue)
            #return the best value and action among all legal actions
            return bestValue, bestAction

        def prune(state, depth, agent, alpha=float("-inf"), beta=float("inf")):
            # expand to next depth 
            if agent == state.getNumAgents(): 
                depth += 1
                agent = 0

            # reached leaf. return the value at node
            if isLeaf(self, state, depth, agent):  
                return self.evaluationFunction(state), None

            #else recursively find out the value of next depth
            return getAlphaBeta(state, depth, agent, alpha, beta)

        #propagate best actions by start pruning the initial game state and return the best actions.
        _,action = prune(gameState, 0, 0)

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        def expectimax(state, depth, agent):
            #expand to next depth nodes as same agent and recurse down if agent is the first one
            if agent == state.getNumAgents():  
                return expectimax(state, depth + 1, 0) 
            
            #return value of node once reaching the leaf
            if isLeaf(self, state, depth, agent):
                return self.evaluationFunction(state)  

            #recursively calculate expectimax for 
            successors = set()
            for action in state.getLegalActions(agent):
                successors.add(expectimax(state.generateSuccessor(agent, action), depth, agent + 1))

            # if agent is a maximizer, return the best moves among the successors
            if isMaximizer(state, agent):
                return max(successors)

            # else, give equal chance for every move and find the average of all possible moves.
            else:
                return sum(successors)/len(successors)

        # of the actions, return max based on the expectimax score on each action
        return max(gameState.getLegalActions(0), key = lambda x: expectimax(gameState.generateSuccessor(0, x), 0, 1))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #weight of food, ghost, and vulnerable ghost to propagate most favorable reward to decide next action
    food = 10.0
    ghost = 10.0
    vulnerableGhost = 100.0

    currentScore = currentGameState.getScore()

    #ghost
    for nextGhost in newGhostStates:
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())

        if ghostDistance > 0:
            #if ghost is scared, positive reward based on weight/distance
            if nextGhost.scaredTimer > 0:  
                currentScore += vulnerableGhost / ghostDistance

            #if ghost is not scared, negative reward based on weight/distance
            else:  
                currentScore -= ghost / ghostDistance

    #food
    #positive reward based on weight/distance of closest food
    foodDistance = [manhattanDistance(newPos, x) for x in newFood.asList()]

    if len(foodDistance) > 0:
        currentScore += food / min(foodDistance)

    return currentScore


def arcConsistencyCSP(csp):

    """
    Implement AC3 here
    """
    "*** YOUR CODE HERE ***"
    #returns neighboring variables of variable var in graph except for variable j
    def getNeighborsExcept(csp, var, j):
        neighbors = []
        for arc in csp.graph:
            x = arc[0]
            y = arc[1]
            if  x == var:
                if y != j:
                  neighbors.append(y)
            elif y == var:
                if x != j:
                   neighbors.append(x)
        return neighbors

    #revise function removes domain value of either xi or xj that does not satisfy the constraint between xi and xj
    def revise(csp, xi, xj):
        revised = 0
        
        for x in csp.domains[xi].copy():
            satisfied = False           
            for y in csp.domains[xj]:

                if x != y:
                    satisfied = True
            if not satisfied:
                csp.domains[xi].remove(x)
                revised = 1

        if revised == 0:
            for x in csp.domains[xj].copy():
                satisfied = False           
                for y in csp.domains[xi]:

                    if x != y:
                        satisfied = True
                if not satisfied:
                    csp.domains[xj].remove(x)
                    revised = 2

        return revised

    #initialize queue with graph (consisting of edges) of csp
    queue = util.Queue()
    graph = list(csp.graph)

    queue.list = graph[::-1]

    while not queue.isEmpty():
        arc = queue.pop()
        revised = revise(csp, arc[0], arc[1])
        #if revised = 1, revise function revised domain of arc[0], so add neighbors of arc[0] to queue
        if revised == 1:
            if len(csp.domains[arc[0]]) == 0:
                return dict()
            neighbors = getNeighborsExcept(csp, arc[0], arc[1])
            
            for xk in neighbors:
                queue.push([xk, arc[0]])
    
        #if revised = 1, revise function revised domain of arc[1], so add neighbors of arc[1] to queue
        elif revised == 2:
            #if fails, return empty dictionary.
            if len(csp.domains[arc[1]]) == 0:
                return dict()
            #
            neighbors = getNeighborsExcept(csp, arc[1], arc[0])
            for xk in neighbors:
                queue.push([xk, arc[1]])
    
    #once queue is empty, meaning there was no more revision to be done, ac-3 is done and return the resultant domain.
    return csp.domains

  

# Abbreviation
better = betterEvaluationFunction
