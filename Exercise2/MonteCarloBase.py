#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor = 0.99, epsilon = 1.0, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.initDiscountFactor = discountFactor
		self.epsilon = epsilon
		self.initEpsilon = epsilon
		# Set dict to init Q-Values of all state-action pairs to zero prior to training
		self.Q = defaultdict(float)

		# To access list of posible actions: Agent.possibleActions


	def learn(self):
		# if state-action pair in dict
		# 
		# 
		# should return complete Q-value table of all states
		# should return Q-value estimate after update of the states you've 
		# encountered in the episode ordered by their first 
		# time appearance in the episode
		self.Qestimate = defaultdict(float) #Empty for now!
		self.Qcomplete = defaultdict(float) #Empty for now!
		return self.Qcomplete, self.Qestimate 
		#raise NotImplementedError

	def toStateRepresentation(self, state):
		self.state = state
		# receive a state
		# output to form suitable to my implementation
		return self.state
		#raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		# Use this to set these data to prepare your agent to learn
		self.currentState = state
		self.actionTaken = action
		self.reward = reward
		self.status = status
		self.nextState = nextState
		#raise NotImplementedError

	def setState(self, state):
		hello = state
		#raise NotImplementedError

	# Use to "reset some states of the agent"
	def reset(self):
		# Also reset Q-values?
		self.discountFactor = self.initDiscountFactor
		self.epsilon = self.initEpsilon
		#raise NotImplementedError

	def act(self): # 
	
		self.action = 'KICK'
		# should return a dictionary of the action that should be taken at every state
		# Or should it be a list of S, A, R until end of episode?
		# or given a state select an action based on epsilon and other parameters?
		return self.action
		#raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		return self.epsilon
		#raise NotImplementedError


	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# given a certain number of actions taken and number of episodes
		# How does epsilon change?
		# probability of taking an action = Epsilon divided across all actions
		# should return a tuple indicating the epsilon used at a certain timestep
		return self.epsilon
		#raise NotImplementedError

# DO NOT CHANGE!
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(0.99, 	1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0

	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset() # get init state from env
		print('Obs:', observation)
		status = 0

		# loop until end of episode(status = 1)
		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon) 
			# makes copy of list
			obsCopy = observation.copy() 
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			# get feedback from environment after taking that action
			nextObservation, reward, done, status = hfoEnv.step(action)
			# store experience 
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			# reset observation for next state
			observation = nextObservation

		agent.learn() 
