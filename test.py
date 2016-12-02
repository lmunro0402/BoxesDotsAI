import DeepNN as NN
import numpy as np
def main():
	AI= NN.Net(12, [2, 3, 12], 2)
	# AI.loadWeights()
	print AI.getWeights()
	print " ----------------------------------- "
	print AI.getMove()
	y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	y = np.asarray(y).reshape(12, 1)
	AI.train(1, y)

if __name__=='__main__':
	main()