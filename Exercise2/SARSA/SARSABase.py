#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict
import numpy as np
import operator

# TODO: fix variable names!(action, state)
# Do not use numtaken actions as iterator for learn!

class SARSAAgent(Agent):
	def __init__(self, learningRate = 0.1, discountFactor = 0.99, epsilon = 1, initVals = 0.0):
		super(SARSAAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon

		self.timeStepEpisode = 0
		self.Q = defaultdict(float)

		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

		# Set to true to print for debugging
		self.P = False

	def reset(self):
		self.timeStepEpisode = 0 

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.numTakenActions = numTakenActions
		self.epsilon = self.e_range[episodeNumber]
		return self.learningRate, self.epsilon	
	
	def setEpsilon(self, epsilon):
		return self.epsilon	

	def setLearningRate(self, learningRate):
		return self.learningRate

	def setState(self, state):
		self.currentState = tuple(state)
		return self.currentState

	def toStateRepresentation(self, state):
		self.state = state
		return self.state

	def act(self):

		# Current timeStep and state
		self.timeState = (self.timeStepEpisode ,self.currentState)
		# Find action with highest Q value given state and timeStep:
		# Make subdict 'S' with all same State and timestep as in 'pair'
		S = defaultdict(float)
		# if Q not empty
		if self.P: print("Q_act= ", self.Q)
		if self.Q:
			for item in self.Q:
				if item[0:2] == self.timeState:
					S[item] = self.Q[item]
			if S: # if S not empty
			# Get KEY of max value from dict 'S'. get only second item from tuple in KEY, which is action
				if self.P: print("S: ",S)
				optimalAction = max(S.items(), key=operator.itemgetter(1))[0][2]
				if self.P: print('Optimal action from S: ', optimalAction)
			else:
				optimalAction = np.random.choice(self.possibleActions, 1)[0]
				if self.P: print('S is empty, acting randomly')
		else:
			# if Q empty, act randomly(first step only)
			optimalAction = np.random.choice(self.possibleActions, 1)[0]
			if self.P: print('Q is empty, acting randomly')

		p = np.random.uniform(0,1,1)
		if p > self.epsilon:
			# Pick best action
			self.action = optimalAction
			if self.P: print("Optimal action taken: ", self.action)
		else:
			# Act randomly
			self.action = np.random.choice(self.possibleActions, 1)[0]
			if self.P: print("Exploring: ", self.action)

		return self.action

	def setExperience(self, state, action, reward, status, nextState):
		# Store experience
		self.state = tuple(state)
		self.action = action
		self.reward = reward
		if type(nextState) == str:
			self.nextState = nextState
		else:
			self.nextState = tuple(nextState)
		if self.P: print('nextState = ',self.nextState)

		self.timeStepEpisode += 1 

	def learn(self):

		# Choose next action
		# Use next action to get Q(S',A')
		self.nextTimeState = (self.timeStepEpisode + 1 ,self.nextState)
		S = defaultdict(float)
		
		if self.P: print("Q_learn= ", self.Q)
		if self.Q: # if Q not empty
			for item in self.Q:
				# If subitems in Q matches nextTimeState
				if item[0:2] == self.nextTimeState:
					# Add entry to S dict
					S[item] = self.Q[item]
			if S: # if S not empty
				# Get KEY of max value from dict 'S'. get only second item from tuple in KEY, which is action
				# Return action of max value from sublist S
				optimalAction = max(S.items(), key=operator.itemgetter(1))[0][2]
			else:
				# If self is empty: act randomly
				optimalAction = np.random.choice(self.possibleActions, 1)[0]
		else:
			# if Q empty, act randomly(first step only)
			optimalAction = np.random.choice(self.possibleActions, 1)[0]

		p = np.random.uniform(0,1,1)
		if p > self.epsilon:
			# Pick best action
			self.nextAction = optimalAction
		else:
			# Act randomly
			self.nextAction = np.random.choice(self.possibleActions, 1)[0]
			
		value = self.Q[self.timeStepEpisode, self.currentState, self.action]
		nextValue = self.Q[self.timeStepEpisode + 1, self.nextState, self.nextAction]
		valueAfterUpdate = value + self.learningRate*(self.reward + self.discountFactor*nextValue - value)
		self.Q[self.timeStepEpisode, self.currentState, self.action] = valueAfterUpdate

		# (Q(s_t,a_t)(t+1) - Q(s_t,a_t)(t))
		self.updateChange = valueAfterUpdate - value
		return self.updateChange

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			print('START')
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			#print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			# skip agent.learn() on first iteration
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation
			print('END')

		#agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
