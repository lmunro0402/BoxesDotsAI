import DeepNN as NN
import numpy as np
def main():
	AI= NN.Net(12, [30, 20, 12], 2)
	# AI.loadWeights()
	weights = AI.getWeights()
	for w in weights:
		print w
	print " ----------------------------------- "
	print AI.getMove()
	y = [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
	y = np.asarray(y).reshape(12, 1)
	AI.train(2, y)

if __name__=='__main__':
	main()