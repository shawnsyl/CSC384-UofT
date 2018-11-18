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


from util import manhattanDistance
from game import Directions
import random, util
import sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        '''
        Priorities:
        1. Get rid of all food (any number of food left is bad, pick a move that reduces food number)
        2. Get closer to food
        2. Care about ghost when we are close enough (wayy worse if you get closer, use reciprocal)
        3. Get super food (better if you get closer, but don't care as much)
        '''
        closestGhost = min(manhattan_distance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
        newFoodList = [food for food in newFood.asList() if food]
        closestFood = min(manhattan_distance(newPos, food) for food in newFoodList) if newFoodList else 0
        shortestScaredTime = min(newScaredTimes)

        numFoodScore = -23*len(newFoodList)
        closestFoodScore = -0.5*(closestFood + 1)
        closestGhostScore = -20/closestGhost if closestGhost > 0 else -99999
        if shortestScaredTime > 0:
            closetGhostScore = 0.5/(closestGhost + 1)
        return numFoodScore + closestFoodScore + closestGhostScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

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

    def terminal(self, state, depth, index):
        # minimax should terminate if we won/lost OR the max player hit max depth
        return (state.isWin() or state.isLose()) or (depth == self.depth and index % state.getNumAgents() == 0)
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
        """
        "*** YOUR CODE HERE ***"
        value, bestMove = self.minimax(gameState, self.index, 0)
        return bestMove

    def minimax(self, state, index, depth):
        bestMove = None
        if self.terminal(state, depth, index):
            return self.evaluationFunction(state), bestMove

        if index % state.getNumAgents() == 0: nextDepth, value = depth + 1, -999999
        else: nextDepth, value = depth, 999999

        for action in state.getLegalActions(index):
            nextPos = state.generateSuccessor(index, action)
            nextVal, nextMove = self.minimax(nextPos, (index + 1) % state.getNumAgents(), nextDepth)
            if index % state.getNumAgents() == 0 and nextVal > value:
                value, bestMove = nextVal, action
            if index % state.getNumAgents() > 0 and nextVal < value:
                value, bestMove = nextVal, action

        return value, bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, bestMove = self.alphabeta(gameState, self.index, 0, -999999, 999999)
        return bestMove

    def alphabeta(self, state, index, depth, alpha, beta):
        bestMove = None
        if self.terminal(state, depth, index):
            return self.evaluationFunction(state), bestMove

        if index % state.getNumAgents() == 0:
            nextDepth, value = depth + 1, -999999
        else:
            nextDepth, value = depth, 999999

        for action in state.getLegalActions(index):
            nextPos = state.generateSuccessor(index, action)
            nextVal, nextMove = self.alphabeta(nextPos, (index + 1) % state.getNumAgents(), nextDepth, alpha, beta)
            if index % state.getNumAgents() == 0: #ALPHA
                if nextVal > value: value, bestMove = nextVal, action
                if value >= beta: return value, bestMove
                alpha = max(alpha, value)
            if index % state.getNumAgents() > 0: #BETA
                if nextVal < value: value, bestMove = nextVal, action
                if value <= alpha: return value, bestMove
                beta = min(beta, value)

        return value, bestMove

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
        value, bestMove = self.expectimax(gameState, self.index, 0)
        return bestMove

    def expectimax(self, state, index, depth):
        bestMove = None
        if self.terminal(state, depth, index):
            return self.evaluationFunction(state), bestMove

        if index % state.getNumAgents() == 0: nextDepth, value = depth + 1, -999999
        else: nextDepth, value = depth, 999999

        for action in state.getLegalActions(index):
            nextPos = state.generateSuccessor(index, action)
            nextVal, nextMove = self.expectimax(nextPos, (index + 1) % state.getNumAgents(), nextDepth)
            if index % state.getNumAgents() == 0 and nextVal > value:
                value, bestMove = nextVal, action
            if index % state.getNumAgents() > 0:
                value = value + (float)(nextVal/(len(state.getLegalActions(index))))

        return value, bestMove

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"
    '''
    Priorities:
    1. Get rid of all food (any number of food left is bad, pick a move that reduces food number)
    2. Get closer to food
    2. Care about ghost when we are close enough (wayy worse if you get closer, use reciprocal)
    3. Get super food (better if you get closer, but don't care as much)
    '''
    closestGhost = min(manhattan_distance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
    newFoodList = [food for food in newFood.asList() if food]
    closestFood = min(manhattan_distance(newPos, food) for food in newFoodList) if newFoodList else 0
    shortestScaredTime = min(newScaredTimes)

    numFoodScore = -len(newFoodList) if len(newFoodList) != 0 else 9999999999
    closestFoodScore = -0.5 * (closestFood + 1)
    closestGhostScore = -3.0 / (closestGhost + 1) if shortestScaredTime == 0 else 0.5 / (closestGhost + 1)
    scaryScore = shortestScaredTime * 0.5
    gameScore = currentGameState.getScore() * 0.6

    return numFoodScore + closestFoodScore + closestGhostScore + scaryScore + gameScore

# Abbreviation
better = betterEvaluationFunction

