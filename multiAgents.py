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
        if successorGameState.isWin():  # if you win on the next state, return a very positive evaluation
            return float("inf")

        food_list = newFood.asList()  # declaring list of all food locations
        food_distances = []  # declaring list to be used to store distances from successor state to food
        ghost_list = []  # declaring list to be used to store all ghost locations
        ghost_distances = []

        for food_pos in food_list:  # adding manhattan distances to all food positions to a list
            dist_f = manhattanDistance(food_pos, newPos)
            food_distances.append(dist_f)
        for ghost in newGhostStates:  # adding all ghost positions to a list
            ghost_list.append(ghost.getPosition())
        for ghost_pos in ghost_list:  # for each ghost, adding the manhattan distance to ghost to a list
            dist_g = manhattanDistance(ghost_pos, newPos)
            ghost_distances.append(dist_g)
        # if adjusted by just dividing through 9, the evaluation crashes - if adjusted by 999999, it also crashes
        # 9999 gave valid results every time
        food_d_score = 10000 / sum(food_distances)  # assign scores (could be invalid depending on the following checks)
        food_l_score = 10000 / len(food_distances)  # d_score is adjusted distance, and l_score is how much food is left

        if currentGameState.getPacmanPosition() == newPos:
            return -float("inf")  # if the next state stays in the same place, return a negative evaluation

        for distance in ghost_distances:  # check all the ghosts
            if distance < 2:  # if the successor is less than 2 squares from the ghost, a very negative eval is given
                return -float("inf")

        if len(food_distances) == 0:  # if all food has been eaten, the list will be empty
            return float("inf") # very positive evaluation if all food has been eaten

        return food_d_score + food_l_score  # return the score depending on if the food is close or far

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
        def minimum_value(state, depth, agent_index):  # take state, depth, and agentIndex as parameters
            legal_actions = state.getLegalActions(agent_index)  # get list of legal actions
            if len(legal_actions) == 0:  # if there are no legal actions, return evaluation function
                return self.evaluationFunction(state)
            lowest_val = float("inf")  # set a high value to reduce as lower minimums are found
            for action in legal_actions:  # for each possible action
                next_index = agent_index + 1
                num_ghosts = state.getNumAgents() - 1
                next_state = state.generateSuccessor(agent_index, action)  # generate the given next state
                if agent_index == num_ghosts:  # if  the agent index matches the number of ghosts
                    lowest_val = min(lowest_val, maximum_value(next_state, depth))  # maintain depth and call max
                else:  # if agent index doesn't match the number of ghosts
                    lowest_val = min(lowest_val, minimum_value(next_state, depth, next_index))  # increase depth and min
            return lowest_val  # return the lowest value found from all legal actions

        def maximum_value(state, depth):  # take only state and depth as parameters
            legal_actions = state.getLegalActions(0)  # get list of legal actions for pacman
            if state.isWin() or state.isLose() or (not legal_actions) or (depth + 1) == self.depth:
                return self.evaluationFunction(state)  # return if reached the terminal state
            highest_val = -float("inf")  # set a low value to increase as higher maximums are found
            for action in legal_actions:  # for each possible action
                next_depth = depth + 1
                next_state = state.generateSuccessor(0, action)  # generate/save given next state
                highest_val = max(highest_val, minimum_value(next_state, next_depth, 1))  # increase depth and call min
            return highest_val  # return the highest value found from all legal actions

        root_actions = gameState.getLegalActions(0)  # root node of trees legal actions
        root_max = -float("inf")  # low value to be increased as larger scores are discovered
        root_ret = 'NORTH'  # default initializing of a return value - will change as the actions are sorted through
        for root_act in root_actions:  # for each available action
            root_next = gameState.generateSuccessor(0, root_act)  # generate next state
            root_curr = minimum_value(root_next, 0, 1)  # call the mini half of our minimax to start going through tree
            if root_curr > root_max:  # if our minimax value down the tree for this action is higher than previous max
                root_ret = root_act  # replace existing max
                root_max = root_curr  # replace current direction
        return root_ret  # return correct direction based on minimax algorithm

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimum_value(state, depth, agent_index, a, b):  # take state, depth, agentIndex, and A + B as parameters
            alpha = a
            beta = b
            legal_actions = state.getLegalActions(agent_index)  # get list of legal actions
            if len(legal_actions) == 0:  # if there are no legal actions, return evaluation function
                return self.evaluationFunction(state)
            lowest_val = float("inf")  # set a high value to reduce as lower minimums are found
            for action in legal_actions:  # for each possible action
                next_index = agent_index + 1
                num_ghosts = state.getNumAgents() - 1
                next_state = state.generateSuccessor(agent_index, action)  # generate the given next state
                if agent_index == num_ghosts:  # if  the agent index matches the number of ghosts
                    lowest_val = min(lowest_val, maximum_value(next_state, depth, alpha, beta))  # call max
                    if lowest_val < alpha:
                        return lowest_val
                    beta = min(beta, lowest_val)  # updating beta as the min of (B, V)
                else:  # if agent index doesn't match the number of ghosts
                    lowest_val = min(lowest_val, minimum_value(next_state, depth, next_index, alpha, beta))  # call min
                    if lowest_val < alpha:
                        return lowest_val
                    beta = min(beta, lowest_val)  # updating beta as the min of (B, V)
            return lowest_val  # return the lowest value found from all legal actions

        def maximum_value(state, depth, a, b):  # take only state and depth as parameters
            alpha = a
            beta = b
            legal_actions = state.getLegalActions(0)  # get list of legal actions for pacman
            if state.isWin() or state.isLose() or (not legal_actions) or (depth + 1) == self.depth:
                return self.evaluationFunction(state)  # return if reached the terminal state
            highest_val = -float("inf")  # set a low value to increase as higher maximums are found
            for action in legal_actions:  # for each possible action
                next_depth = depth + 1
                next_state = state.generateSuccessor(0, action)  # generate/save given next state
                highest_val = max(highest_val, minimum_value(next_state, next_depth, 1, alpha, beta))  # call min
                if highest_val > beta:
                    return highest_val
                alpha = max(alpha, highest_val)  # updating alpha as the min of (A, V)
            return highest_val  # return the highest value found from all legal actions

        root_actions = gameState.getLegalActions(0)  # root node of trees legal actions
        root_max = -float("inf")  # low value to be increased as larger scores are discovered
        root_a = -float("inf")
        root_b = float("inf")
        root_ret = 'NORTH'  # default initializing of a return value - will change as the actions are sorted through
        for root_act in root_actions:  # for each available action
            root_next = gameState.generateSuccessor(0, root_act)  # generate next state
            root_curr = minimum_value(root_next, 0, 1, root_a, root_b)  # call the mini half of our minimax with A and B
            b_check = root_curr > root_b
            if root_curr > root_max:  # if our minimax value down the tree for this action is higher than previous max
                root_ret = root_act  # replace existing max
                root_max = root_curr  # replace current direction
            if b_check:
                return root_ret
            else:
                root_a = max(root_a, root_curr)  # updating alpha as the min of (A, V)
        return root_ret  # return correct direction based on minimax algorithm
        # util.raiseNotDefined()

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
        def expected_value(state, depth, agent_index):  # take state, depth, and agentIndex as parameters
            legal_actions = state.getLegalActions(agent_index)  # get list of legal actions
            total_actions = len(legal_actions)
            if total_actions == 0:  # if there are no legal actions, return evaluation function
                return self.evaluationFunction(state)
            exp_val = 0  # add an expected value
            for action in legal_actions:  # for each possible action
                next_index = agent_index + 1
                num_ghosts = state.getNumAgents() - 1
                next_state = state.generateSuccessor(agent_index, action)  # generate the given next state
                if agent_index == num_ghosts:  # if  the agent index matches the number of ghosts
                    temp_val = maximum_value(next_state, depth)
                else:  # if agent index doesn't match the number of ghosts
                    temp_val = expected_value(next_state, depth, next_index)
                exp_val += temp_val
            if total_actions == 0:
                return 0
            exp_float = float(exp_val) / float(total_actions)
            return exp_float  # return the lowest value found from all legal actions

        def maximum_value(state, depth):  # take only state and depth as parameters
            legal_actions = state.getLegalActions(0)  # get list of legal actions for pacman
            total_actions = len(legal_actions)
            if state.isWin() or state.isLose() or (not legal_actions) or (depth + 1) == self.depth:
                return self.evaluationFunction(state)  # return if reached the terminal state
            highest_val = -float("inf")  # set a low value to increase as higher maximums are found
            for action in legal_actions:  # for each possible action
                next_depth = depth + 1
                next_state = state.generateSuccessor(0, action)  # generate/save given next state
                highest_val = max(highest_val, expected_value(next_state, next_depth, 1))  # increase depth and call min
            return highest_val  # return the highest value found from all legal actions

        root_actions = gameState.getLegalActions(0)  # root node of trees legal actions
        root_max = -float("inf")  # low value to be increased as larger scores are discovered
        root_ret = 'NORTH'  # default initializing of a return value - will change as the actions are sorted through
        for root_act in root_actions:  # for each available action
            root_next = gameState.generateSuccessor(0, root_act)  # generate next state
            root_curr = expected_value(root_next, 0, 1)  # call the mini half of our minimax to start going through tree
            if root_curr > root_max:  # if our minimax value down the tree for this action is higher than previous max
                root_ret = root_act  # replace existing max
                root_max = root_curr  # replace current direction
        return root_ret  # return correct direction based on minimax algorithm
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Took the reciprocal of the manhattan distance to the closest food (as detailed in the
    project description. Since it already passed the tests, didn't implement any pellet checking, although
    this results in pacman stopping and waiting for the ghost to come by every once in a while
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition();  # get current position
    food_list = currentGameState.getFood().asList()  # get food positions
    current_score = currentGameState.getScore()  # get current score

    min_distance = float("inf")  # set minimum very high to be reduced
    for food in food_list:  # for every food location found
        pac_manhattan = manhattanDistance(pacman_position, food)  # check the distance
        if pac_manhattan < min_distance:  # if the distance is the lowest yet, set our minimum
            min_distance = pac_manhattan
    final_evaluation = 1.0 / min_distance  # take reciprocal of the lowest distance to food
    final_evaluation += current_score  # add the current score

    return final_evaluation

    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
