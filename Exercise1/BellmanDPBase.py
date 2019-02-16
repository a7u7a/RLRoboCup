from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self):
		self.MDP = MDP()

	def initVs(self):
		raise NotImplementedError

	# method must return a tuple of (values, policy)
	def BellmanUpdate(self):
		a = 10
		b = 2
		return(a,b)

		#raise NotImplementedError
		

if __name__ == '__main__':
	solution = BellmanDPSolver()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)