#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict
import numpy as np

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor = 0.99, epsilon = 1.0, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.initDiscountFactor = discountFactor
		self.epsilon = epsilon
		# Set init epsilon for reset purposes
		self.initEpsilon = epsilon
		# G = dict, where KEY= tuple state, action and VALUE= list of rewards 
		self.episodeStateActions = {}
		self.rewards = []
		
		self.Q = {}

		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

	def toStateRepresentation(self, state):
		self.state = state
		# receive a state
		# output to form suitable to my implementation
		return self.state
		#raise NotImplementedError

	def learn(self):
		for item in self.episodeStateActions.keys():
			self.Q[item] = np.average(self.rewards[self.episodeStateActions[item]:])


		# if state-action pair in dict
		# 
		# 
		# should return complete Q-value table of all states
		# should return Q-value estimate after update of the states you've 
		# encountered in the episode ordered by their first 
		# time appearance in the episode
		return self.Q
		#raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		# Use this to set these data to prepare your agent to learn
		self.currentState = state
		self.actionTaken = action
		self.reward = reward
		self.status = status
		self.nextState = nextState

		# TODO: Add discount factor here somewhere!!
		# Create tuple with state-action pair
		pair = (tuple(self.currentState), self.actionTaken)
		# If this is the first time S,A pair is seen, add to dict and note timestep
		if pair not in self.episodeStateActions.keys():
			self.episodeStateActions[pair] = self.timeStep
		
		# Append reward to list of rewards
		self.rewards += [reward]
		
		

		#raise NotImplementedError

	def setState(self, state):
		hello = state
		#raise NotImplementedError

	# Use to "reset some states of the agent"
	def reset(self):
		# Also reset Q-values?
		self.discountFactor = self.initDiscountFactor
		self.epsilon = self.initEpsilon
		self.episodeStateActions = {}
		#raise NotImplementedError

	def act(self): # 
		# should return the action that should be taken at every state
		# calculate probability of selecting greedy or not
		#nonGreedy = (self.epsilon/len(self.possibleActions))

		# TODO: Find optimal action for state according to Q!

		p = np.random.uniform(0,1,1)
		if p > self.epsilon:
			# Pick best action
			self.action = "KICK" # Change!
			#print('GREEDY!: ',self.epsilon)
		else:
			# Act randomly
			self.action = np.random.choice(self.possibleActions, 1)[0]
			#print('RANDOMLY!: ',self.epsilon)
	
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

		# get timestep
		self.timeStep = numTakenActions
		# draw from range
		self.epsilon = self.e_range[episodeNumber]
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
		# get init state from env
		observation = hfoEnv.reset() 
		status = 0

		# loop until end of episode(status = 1)
		# 1 loop = 1 state/step
		while status==0:
			# compute epsilon for this iteration
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			# Set epsilon (??)
			agent.setEpsilon(epsilon) 
			# makes copy of observation(list)
			obsCopy = observation.copy() 
			# convert to own representation
			agent.setState(agent.toStateRepresentation(obsCopy))
			# choose action
			action = agent.act()
			numTakenActions += 1
			# get feedback from environment after taking that action
			nextObservation, reward, done, status = hfoEnv.step(action)
			# store experience 
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			# reset observation for next state
			observation = nextObservation

		agent.learn()
		print('Q: ',agent.Q) 
