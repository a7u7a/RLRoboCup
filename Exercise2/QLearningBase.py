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
		self.Q = defaultdict(float)
		self.timeStepEpisode = 0

		# List to schedule epsilon values from (assumes 5000 episodes)
		X = np.linspace(0.01, 0.05, 5000, endpoint=True)
		# Asymptote schedule for e-soft
		self.e_range = (1/X**2)
		# Normalize to fit range 0.0-1.0
		self.e_range = (self.e_range-min(self.e_range))/(max(self.e_range)-min(self.e_range))

		# Set to true to print for debugging
		self.P = True

	def reset(self):
		self.timeStepEpisode = 0
		
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
		self.currentState = tuple(state)

	def toStateRepresentation(self, state):
		self.state = state
		return self.state

	def act(self):
		# Choose action from state using policy derived from Q
		# find matching state, select action with max value e-greedy

		S = defaultdict(float)

		if self.Q: # if Q not empty
			for item in self.Q:
				if item[0] == self.state:
					S[item] = self.Q[item]
			if S: # if S not empty
				greedyAction = max(S.items(), key=operator.itemgetter(1))[0][1]
				if self.P: print('greedyAction(act)= ', greedyAction)
			else: # if self is empty: act randomly
				greedyAction = np.random.choice(self.possibleActions, 1)[0]
		else: # if Q empty, act randomly
			greedyAction = np.random.choice(self.possibleActions, 1)[0]
		
		p = np.random.uniform(0,1,1)
		if p > self.epsilon:
			# Pick best action
			self.action = greedyAction
			if self.P: print("Greedy action taken: ", self.action)
		else:
			# Act randomly
			self.action = np.random.choice(self.possibleActions, 1)[0]
			if self.P: print("Exploring: ", self.action)
		
		return self.action

	def setExperience(self, state, action, reward, status, nextState):
		self.currentState = tuple(state)
		self.action = action
		if type(nextState) == str:
			self.nextState = nextState
		else:
			self.nextState = tuple(nextState)
		self.timeStepEpisode += 1
		self.reward = reward

	def learn(self):
		# Improve the policy using update 
		pair = (self.currentState, self.action)
		nextPair = ()

		S = defaultdict(float)

		if self.Q: # if Q not empty
			for item in self.Q:
				if item[0] == self.nextState: # find matching state in Q 
					S[item] = self.Q[item]
			if S: # if S not empty
				maxV = max(S.items(), key=operator.itemgetter(1))[1]
				if self.P: print('maxA(learn)= ', maxV)
			else:
				maxV = 0
		else:
			maxV = 0

		valueAfterUpdate = self.Q[pair]
		valueBeforeUpdate = valueAfterUpdate + self.learningRate*(self.reward+self.discountFactor*maxV-valueAfterUpdate)
		self.Q[pair] = valueBeforeUpdate

		return valueAfterUpdate


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
			print('valueAfterUpdate: ',agent.learn())
			
			observation = nextObservation
	
