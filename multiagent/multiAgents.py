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

import random
import util

from game import Agent
from game import Directions
from util import manhattanDistance


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # print(currentGameState,action,successorGameState,newPos,newFood,newGhostStates,newScaredTimes)

        minGhost = 9999
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                dist = manhattanDistance(list(map(int, ghostState.getPosition())), newPos)
                if dist < minGhost:
                    minGhost = dist
        minFood = 1000
        for foodPos in newFood.asList():
            dist = manhattanDistance(foodPos, newPos)
            if dist < minFood:
                minFood = dist
        return successorGameState.getScore() - minFood / 3 - 7 / (minGhost + 1)
        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinMaxValue:

    def __init__(self, score, action):
        self.score = score
        self.action = action


def gt(self, other):
    return self.score > other


def lt(self, other):
    return self.score < other


def getNextArgs(agentIndex, numAgents, depth):
    if agentIndex == numAgents - 1:
        return 0, depth - 1
    else:
        return agentIndex + 1, depth


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
        return self.getMiniMaxValue(gameState, 0, self.depth).action

    def getMiniMaxValue(self, gameState, AgentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return MinMaxValue(self.evaluationFunction(gameState), Directions.STOP)
        else:
            maxValue = 100000
            if AgentIndex != 0:
                return self.getValue(gameState, AgentIndex, depth, lt, maxValue)
            else:
                return self.getValue(gameState, AgentIndex, depth, gt, -maxValue)

    def getValue(self, gameState, agentIndex, depth, minmaxFnc, initValue):
        nextAgent, nextDept = getNextArgs(agentIndex, gameState.getNumAgents(), depth)
        score = initValue
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            currValue = self.getMiniMaxValue(gameState.generateSuccessor(agentIndex, action),
                                             nextAgent, nextDept)
            if minmaxFnc(currValue, score):
                score = currValue.score
                bestAction = action
        return MinMaxValue(score, bestAction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = 100000
        return self.getMiniMaxValue(gameState, 0, self.depth, -maxValue, maxValue).action

    def getMiniMaxValue(self, gameState, AgentIndex, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return MinMaxValue(self.evaluationFunction(gameState), Directions.STOP)
        else:
            maxValue = 100000
            if AgentIndex != 0:
                return self.getValue(gameState, AgentIndex, depth, lt, maxValue, alpha, beta)
            else:
                return self.getValue(gameState, AgentIndex, depth, gt, -maxValue, alpha, beta)

    def getValue(self, gameState, agentIndex, depth, minmaxFnc, initValue, alpha, beta):
        nextAgent, nextDept = getNextArgs(agentIndex, gameState.getNumAgents(), depth)
        score = initValue
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            currValue = self.getMiniMaxValue(gameState.generateSuccessor(agentIndex, action),
                                             nextAgent, nextDept, alpha, beta)
            if minmaxFnc(currValue, score):
                score = currValue.score
                bestAction = action
            if lt == minmaxFnc:
                if currValue.score < alpha:
                    return MinMaxValue(currValue.score, action)
                beta = min(beta, score)
            else:
                if currValue.score > beta:
                    return MinMaxValue(currValue.score, action)
                alpha = max(alpha, score)
        return MinMaxValue(score, bestAction)


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
        return self.getExpectedValue(gameState, 0, self.depth).action

    def getExpectedValue(self, gameState, AgentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return MinMaxValue(self.evaluationFunction(gameState), Directions.STOP)
        else:
            maxValue = 1000000
            if AgentIndex != 0:
                return self.getValue(gameState, AgentIndex, depth, None, maxValue)
            else:
                return self.getValue(gameState, AgentIndex, depth, gt, -maxValue)

    def getValue(self, gameState, agentIndex, depth, minmaxFnc, initValue):
        nextAgent, nextDept = getNextArgs(agentIndex, gameState.getNumAgents(), depth)
        score = initValue
        bestAction = Directions.STOP
        expSum =0
        for action in gameState.getLegalActions(agentIndex):
            currValue = self.getExpectedValue(gameState.generateSuccessor(agentIndex, action),
                                              nextAgent, nextDept)
            if minmaxFnc is None:
                expSum += currValue.score
            elif minmaxFnc(currValue, score):
                score = currValue.score
                bestAction = action
        if minmaxFnc is None:
            score = expSum / len(gameState.getLegalActions(agentIndex))
        return MinMaxValue(score, bestAction)



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # print(currentGameState,action,successorGameState,newPos,newFood,newGhostStates,newScaredTimes)

    minGhost = 9999
    for ghostState in newGhostStates:
        if ghostState.scaredTimer == 0:
            dist = manhattanDistance(list(map(int, ghostState.getPosition())), position)
            if dist < minGhost:
                minGhost = dist
        else:
            minGhost = -2  # -1 is zero division at the end
    minFood = 1000
    for foodPos in newFood.asList():
        dist = manhattanDistance(foodPos, position)
        if dist < minFood:
            minFood = dist
    return currentGameState.getScore() - minFood / 3 - 7 / (minGhost + 1)


# Abbreviation
better = betterEvaluationFunction
