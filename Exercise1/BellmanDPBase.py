from MDP import MDP
import numpy as np
from plotvalue import plot_value_and_policy

class BellmanDPSolver(object):
	
	def __init__(self, discountRate):
		self.discountRate = discountRate
		self.env = MDP()
		self.initVs()

	def initVs(self):
		# init values and policy as dicts with keys = states and values = 0
		# Values
		self.V = dict.fromkeys(self.env.S, 0)
		# Policy
		self.policy = dict.fromkeys(self.env.S, 0)
		
	def BellmanUpdate(self):
		for s in self.env.S:
			# Init Action - Value dict with Keys = Actions, Values = values(0)
			Av = np.zeros(len(self.env.A))
			c = 0
			for a in self.env.A:
				# get transition probability for current action and state
				t_probs = self.env.probNextStates(s,a).values()
				# sample given transition probability distribution
				sample = np.random.choice(len(t_probs),1,p=tuple(t_probs))[0]
				# get possible next states
				states = tuple(self.env.probNextStates(s,a).keys())
				# next state from sample
				next_state = states[sample]
				# prob. of chosen action
				prob = tuple(t_probs)[sample]
				# fetch reward
				reward = self.env.getRewards(s, a, next_state)
				# add to the expectation values
				Av[c] += prob * (reward + discountRate * self.V[next_state])
				c = c+1
			# find max value and write to dict
			self.V[s] = np.max(Av)
			# compute deterministic greedy policy (action that yields more reward) find
			# index of max action-value in state and write to dict
			self.policy[s] = self.env.A[np.argmax(Av)]
		return(self.V,self.policy)
		# method must return a tuple of (optimal value function, policy)

if __name__ == '__main__':
	discountRate = 1
	solution = BellmanDPSolver(discountRate)
	# run n times
	for i in range(10):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)