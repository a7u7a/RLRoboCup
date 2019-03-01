#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict
import numpy as np
import operator

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
		self.timeStepEpisode = 0
		self.timeStep = 0
		self.pair = ()
		#self.Q = {(((1, 2), (2, 2)), 'DRIBBLE_RIGHT'):0}
		self.Q = defaultdict(float)
		self.valueList = []
		
		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

	def reset(self):
		# Also reset Q-values?
		self.discountFactor = self.initDiscountFactor
		self.valueList = []
		# Reset timestep for new episode
		self.timeStepEpisode = 0
		# Reset rewards to zero
		self.rewards = []
		self.episodeStateActions = {}
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.episodeNumber = episodeNumber

		# draw from range
		self.epsilon = self.e_range[episodeNumber]

		return self.epsilon

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		return self.epsilon		

	def setState(self, state):
		self.currentState = tuple(state)

	def toStateRepresentation(self, state):
		# Keep state representation
		self.state = state
		return self.state

	def act(self): # 
		# Current state given timestep in episode
		self.timeState = (self.timeStepEpisode ,self.currentState)
		#print("timeState: ",self.timeState)
		# Find action with highest Q value given state and timeStep:
		# Make subdict 'S' with all same State and timestep as in 'pair'
		# (Could be improved by usung numpy dataframe?)
		S = defaultdict(float)
		# if Q not empty
		if self.Q:
			for item in self.Q:
				if item[0:2] == self.timeState:
					S[item] = self.Q[item]
			if S:
			# Get KEY of max value from dict 'S'. get only second item from tuple in KEY, which is action
				optimalAction = max(S.items(), key=operator.itemgetter(1))[0][2]
			else:
				optimalAction = np.random.choice(self.possibleActions, 1)[0]
				#print('S is empty, acting randomly')
		else:
			# if Q empty, act randomly(first step only)
			optimalAction = np.random.choice(self.possibleActions, 1)[0]
			#print('Q is empty, acting randomly')

		#print('Q: ',self.Q)

		p = np.random.uniform(0,1,1)
		if p > self.epsilon:
			# Pick best action
			self.action = optimalAction
		else:
			# Act randomly
			self.action = np.random.choice(self.possibleActions, 1)[0]
			#print('RANDOMLY!: ',self.epsilon)
		return self.action

	def setExperience(self, state, action, reward, status, nextState):
		# Use this to set these data to prepare your agent to learn
		self.currentState = tuple(state)
		self.actionTaken = action
		self.status = status
		self.nextState = nextState

		# Create tuple with current timestep and state-action pair
		self.pair = (self.timeStepEpisode ,self.currentState, self.actionTaken)
		#print("pair: ", self.pair)

		# If this is the first time (S,A) pair is seen:
		if self.pair not in self.episodeStateActions:
			# Add to dict and note timestep
			self.episodeStateActions[self.pair] = self.timeStepEpisode

		# Apply discount factor to previous rewards
		self.rewards = [x * self.discountFactor for x in self.rewards]
		# Append new reward to list of rewards
		self.rewards += [reward]

		# Update Q: for every s,a found in episode
		# use timeStepEpisode as slicer index to 
		# average from first time s,a to current time using
		# list of rewards per timestep
		value = np.average(self.rewards[self.episodeStateActions[self.pair]:])
		# append value to valuelist for learn() output
		self.valueList.append(value)

		# What if instead of replacing we add?
		self.Q[self.pair] += value

		# Update timestep counters
		self.timeStepEpisode += 1 
		self.timeStep += 1

	def learn(self):
		#create a tuple for output
		self.completeQ = (self.episodeNumber, self.valueList)
		return self.Q, self.completeQ


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
			#print("start")
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
			#print("end")

		agent.learn()
		#print('completeQ: ',agent.completeQ)
		#print("Q: ", agent.Q)


# output dict with policy and values