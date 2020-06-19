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
        caps = currentGameState.getCapsules()
        "*** YOUR CODE HERE ***"
        score = 0

        curPos = currentGameState.getPacmanPosition()  # current pacman position
        curFood = currentGameState.getFood()  # food map
        FoodList = curFood.asList()  # list with all foods
        ghostPosition = successorGameState.getGhostPositions()

        # Distance between PacMan and Ghost
        distx = abs(newPos[0] - ghostPosition[0][0])
        disty = abs(newPos[1] - ghostPosition[0][1])

        # Distance between PacMan Current Position and Goal Food
        distcx = abs(curPos[0] - FoodList[0][0])
        distcy = abs(curPos[1] - FoodList[0][1])

        # Distance between PacMan New Position and Goal Food
        distnx = abs(newPos[0] - FoodList[0][0])
        distny = abs(newPos[1] - FoodList[0][1])

        # If there is a capsule
        if caps != []:
            # Distance between PacMan Current Position and Capsule
            distccx = abs(curPos[0] - caps[0][0])
            distccy = abs(curPos[1] - caps[0][1])

            # Distance between PacMan New Position and Capsule
            distncx = abs(newPos[0] - caps[0][0])
            distncy = abs(newPos[1] - caps[0][1])

        # If we have a safe move: We are not in danger dying from the ghost
        if distx > 1 or disty > 1:
            if caps != []:
                # Try to get closer to the capsule
                if (distncx < distccx or distncy < distccy):
                    score = score + 2
                elif (distccx == distncx and distccy == distncy):
                    score = score + 1
                elif (distncx > distccx or distncy > distccy):
                    score
            else:
                # Try to get closer to the food
                if (distnx < distcx or distny < distcy):
                    score = score + 2
                elif (distcx == distnx and distcy == distny):
                    score = score + 1
                elif (distnx > distcx or distny > distcy):
                    score
            score = score + 3
        # If we are close to the ghost , try your best to dodge it
        elif (distx == 1 and disty == 1):
            score = score + 2
        elif (distx == 0 and disty == 1):
            score = score + 1
        elif (distx == 1 and disty == 0):
            score = score + 1
        else:
            score = score + 0

        # evaluation results
        return score
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
        self.NumAgents = 0  # number of agents
        self.root = None  # root of the tree, the first currentgamestate
        self.move = None  # best move

    # If state of the game belongs to the root of the tree
    def isRoot(self, state):
        if state == self.root:
            return True
        else:
            return False

    # Checks if we are on a terminal situation
    def TerminalTest(self, state, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return True
        else:
            return False

    # Returns the score of the leaf
    def Utility(self, state):
        return scoreEvaluationFunction(state)

    # Returns the legalactions that agent can do
    def Actions(self, state, agent):
        return state.getLegalActions(agent)

    # Returns the successors of agent , given an action
    def Result(self, state, a, agent):
        return state.generateSuccessor(agent, a)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # Max Player
    def MaxValue(self, state, index, depth):
        if self.TerminalTest(state, depth):
            return self.Utility(state)
        # Initializes index , so we can have multiple min level
        index %= (self.NumAgents - 1)
        v = -100000
        for a in self.Actions(state, index):
            v = max(self.MinValue(self.Result(state, a, index), index + 1, depth), v)
        return v

    # Min Player
    def MinValue(self, state, index, depth):
        if self.TerminalTest(state, depth):
            return self.Utility(state)

        v = 100000
        # For each ghost , we will create a new min level without decreasing the depth
        # unless we pass the turn to max player
        if index + 1 == self.NumAgents:
            for a in self.Actions(state, index):
                v = min(v, self.MaxValue(self.Result(state, a, index), index, depth - 1))
        else:
            for a in self.Actions(state, index):
                v = min(v, self.MinValue(self.Result(state, a, index), index + 1, depth))
        return v

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

        # minimax-decision
        self.NumAgents = gameState.getNumAgents()
        v = -10000
        action = ''
        # The following for will evaluate each action starting from the root , and finds the one that will lead
        # to a maximum possible score
        for a in self.Actions(gameState, self.index):
            score = self.MinValue(self.Result(gameState, a, self.index), self.index + 1, self.depth)
            if score > v:
                action = a
                v = score
        return action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # Max Player implementation as shown on site
    def Max_Value(self, state, a, b, index, depth):
        if self.TerminalTest(state, depth):
            return self.Utility(state)

        v = -100000
        index %= (self.NumAgents - 1)
        for action in self.Actions(state, index):
            score = self.Min_Value(self.Result(state, action, index), a, b, index + 1, depth)
            if score > v:
                v = score
                # We will keep the move , starting from the root that led us to the best score
                if self.isRoot(state):
                    self.move = action
            if v > b:
                return v
            a = max(a, v)
        return v

    # Min Player
    def Min_Value(self, state, a, b, index, depth):
        if self.TerminalTest(state, depth):
            return self.Utility(state)

        v = 1000000
        # Creates the min levels that we need , in order to account each possible node
        # ( like I did in minimax )
        if index + 1 == self.NumAgents:
            for action in self.Actions(state, index):
                v = min(v, self.Max_Value(self.Result(state, action, index), a, b, index, depth - 1))
                if v < a:
                    return v
                b = min(v, b)
            return v
        else:
            for action in self.Actions(state, index):
                v = min(v, self.Min_Value(self.Result(state, action, index), a, b, index + 1, depth))
                if v < a:
                    return v
                b = min(v, b)
            return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.NumAgents = gameState.getNumAgents()
        self.root = gameState
        # alpha-beta-search
        # one call of Max_Value is enough to iterate the whole tree and keep the optimal move
        self.Max_Value(gameState, -10000, 10000, self.index, self.depth)
        return self.move
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Sums the score of each move
    def chance_sum(self, actionlist):
        sum = 0
        for a in actionlist:
            sum += a[0]
        return sum

    def Expectimax(self, state, depth, index=0):
        if self.TerminalTest(state, depth):
            return (self.evaluationFunction(state),)

        # the new depth that we are going to reach with expectimax
        newDepth = None
        if index + 1 == self.NumAgents:
            newDepth = depth - 1
        else:
            newDepth = depth

        # the agent that will grow its nodes
        agent = (index + 1) % self.NumAgents

        # create a list of (value,action) elements , so we can keep the optimal move with the best possible score
        actionlist = [(self.Expectimax(self.Result(state, action, index), newDepth, agent)[0], action) for action in
                      self.Actions(state, index)]

        # number of legal actions for our chance node
        length_actlist = len(self.Actions(state, index))

        # Max Node
        if index == 0:
            return max(actionlist)
        # Chance Node
        else:
            return (self.chance_sum(actionlist) / length_actlist,)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.NumAgents = gameState.getNumAgents()
        # (Value,Move)
        #   ^      ^ Our optimal move that led us to the max -
        #   |------------------------------------------------|
        return self.Expectimax(gameState, self.depth)[1]
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Keeping track of all parameters
    score = currentGameState.getScore()
    curfood = currentGameState.getFood()
    foodlist = curfood.asList()
    curcapsules = currentGameState.getCapsules()
    ghostposition = currentGameState.getGhostPositions()
    pacmanposition = currentGameState.getPacmanPosition()

    # Distance between ghost and pacman
    distx = abs(ghostposition[0][0] - pacmanposition[0])
    disty = abs(ghostposition[0][1] - pacmanposition[1])

    # If we are on a safe zone : we can move freely without being afraid to die
    if distx > 1 or disty > 1:
        score += 30
        foods = []
        caps = []
        # Tries to return the best score , for the closest food
        for f in foodlist:
            dstfx = abs(pacmanposition[0] - f[0])
            dstfy = abs(pacmanposition[1] - f[1])
            # distance factor for pac's - food distance
            foods.append(dstfx + dstfy)
        if foods != []:
            # return score for closest food , which will outtake the other food dots
            score *= (50 - min(foods))
    elif distx == 1 and disty == 1:
        score += 20.0
    elif distx == 0 and disty == 1:
        score += 10.0
    elif distx == 1 and disty == 0:
        score += 10.0
    else:
        score -= 10

    return score
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
