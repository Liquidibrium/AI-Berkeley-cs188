# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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
from learningAgents import ValueEstimationAgent
from mdp import MarkovDecisionProcess
from util import Counter

INITIAL_VALUE = -1000000


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """

        self.mdp = mdp  # type: MarkovDecisionProcess
        self.discount = discount  # type: float
        self.iterations = iterations  # type: int
        # A Counter is a dict with default 0
        self.values = util.Counter()  # type: Counter
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for iteration in range(0, self.iterations):
            new_values = util.Counter()
            for state in states:
                best_value = INITIAL_VALUE
                for action in self.mdp.getPossibleActions(state):
                    curr_value = self.computeQValueFromValues(state, action)
                    if curr_value > best_value:
                        best_value = curr_value
                        new_values[state] = best_value

            self.updateValues(new_values, states)

    def updateValues(self, new_values, states):
        for state in states:
            self.values[state] = new_values[state]

    def getIterativeValue(self, action, next_state, transaction_probability, curr_state):
        return transaction_probability * (self.discount * self.values[next_state] +
                                          self.mdp.getReward(curr_state, action, next_state))

    def getValue(self, state):
        """
            Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionList = self.mdp.getTransitionStatesAndProbs(state, action)
        sum_res = 0
        for next_state, prob in transitionList:
            sum_res += self.getIterativeValue(action, next_state, prob, state)
        return sum_res

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_q_value = INITIAL_VALUE
        res_action = None
        for action in self.mdp.getPossibleActions(state):
            curr_value = self.computeQValueFromValues(state, action)
            if curr_value > best_q_value:
                res_action = action
                best_q_value = curr_value
        return res_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        length = len(states)
        for iteration in range(0, self.iterations):
            state = states[iteration % length]
            if self.mdp.isTerminal(state):
                continue
            best_value = self.getBestQValue(state)
            self.values[state] = best_value

    def getBestQValue(self, state):
        best_value = INITIAL_VALUE
        for action in self.mdp.getPossibleActions(state):
            curr_value = self.computeQValueFromValues(state, action)
            if curr_value > best_value:
                best_value = curr_value
        return best_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = self.compute_predecessors()
        pq = util.PriorityQueue()
        self.update_queue(pq)
        self.iterate(pq, predecessors)

    def iterate(self, pq, predecessors):
        for iteration in range(0, self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getBestQValue(state)

            self.updatePredecessorsInPQ(pq, predecessors, state)

    def updatePredecessorsInPQ(self, pq, predecessors, state):
        for p in predecessors[state]:
            if self.mdp.isTerminal(p):
                continue
            best_value = self.getBestQValue(p)
            diff = abs(self.values[p] - best_value)
            if diff > self.theta:
                pq.update(p, -diff)

    def update_queue(self, pq):
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            best_value = self.getBestQValue(state)
            diff = abs(self.values[state] - best_value)
            pq.update(state, -diff)

    def compute_predecessors(self):
        predecessors = dict()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for next_state, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in predecessors:
                        predecessors[next_state] = set()
                    predecessors[next_state].add(state)
        return predecessors
