#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict
import numpy as np

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon

		self.Q = defaultdict(float)
		self.exp = []

		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

	def reset(self):
		raise NotImplementedError

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.numTakenActions = numTakenActions
		self.epsilon = self.e_range[episodeNumber]
		return self.epsilon	
	
	def setEpsilon(self, epsilon):
		raise NotImplementedError

	def setLearningRate(self, learningRate):
		raise NotImplementedError		

	def setState(self, state):
		self.currentState = state

	def toStateRepresentation(self, state):
		# Keep state representation
		self.state = state
		return self.state

	def act(self):

		# Current timeStep and state
		self.timeState = (self.timeStepEpisode ,self.currentState)
		# Find action with highest Q value given state and timeStep:
		# Make subdict 'S' with all same State and timestep as in 'pair'
		S = defaultdict(float)
		# if Q not empty
		if self.P: print("Q(act): ", self.Q)
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
		self.state = state
		self.action = action
		self.reward = reward
		self.nextState = nextState

		self.exp += [[self.state, self.action, self.reward, self.nextState]]


		

	def learn(self):
		# by using numTakenActions we exclude terminal state
		for item in range(self.numTakenActions):




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
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
