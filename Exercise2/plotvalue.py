import matplotlib.pyplot as plt
import numpy as np

# Currently not showing actions in policy figure!

def plot_value_and_policy(values, policy, iterno):
	
	#path = "/animation/"
	data = np.zeros((5,5))

	width = 1

	plt.figure(figsize = (12, 4))
	plt.subplot(1, 2, 1)
	plt.title('Value')
	for y in range(data.shape[0]):
		for x in range(data.shape[1]):
			data[y][x] = values[(x,y)]
			plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x], horizontalalignment='center',verticalalignment='center',)

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)

	plt.subplot(1, 2, 2)
	plt.title('Policy')
	for y in range(5):
		for x in range(5):
			for action in policy[(x,y)]:
				if action == 'DRIBBLE_UP':
					plt.annotate('',(x+0.5, y),(x+0.5,y+0.5),arrowprops={'width':width})
				if action == 'DRIBBLE_DOWN':
					plt.annotate('',(x+0.5, y+1),(x+0.5,y+0.5),arrowprops={'width':width})
				if action == 'DRIBBLE_RIGHT':
					plt.annotate('',(x+1, y+0.5),(x+0.5,y+0.5),arrowprops={'width':width})
				if action == 'DRIBBLE_LEFT':
					plt.annotate('',(x, y+0.5),(x+0.5,y+0.5),arrowprops={'width':width})
				if action == 'SHOOT':
					plt.text(x + 0.5, y + 0.5, action, horizontalalignment='center',verticalalignment='center',)

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)
	plt.savefig(str(iterno) + '.png')
