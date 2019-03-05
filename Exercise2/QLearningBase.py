#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

from collections import defaultdict
import numpy as np
import operator

class QLearningAgent(Agent):
	def __init__(self, learningRate = 0.1, discountFactor = 0.9, epsilon = 1 , initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon

		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

		# Set to true to print for debugging
		self.P = False

	def reset(self):
		raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.epsilon = self.e_range[episodeNumber]
		return self.learningRate, self.epsilon		

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		return self.epsilon

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate	
		return self.learningRate

	def setState(self, state):
		self.currentState = state

	def toStateRepresentation(self, state):
		self.state = state
		return self.state

	def act(self):
		# Choose action from state using policy derived from Q

		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.currentState = state
		self.action = action
		self.nextState = nextState


	def learn(self):
		raise NotImplementedError


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	
