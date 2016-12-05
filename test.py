import DeepNN as NN
import numpy as np
def main():
	AI= NN.Net(4, [1, 2, 4], 2)
	# AI.loadWeights()
	weights = AI.getWeights()
	for w in weights:
		print w
	print " ----------------------------------- "
	print AI.getMove()
	y = [1, 0, 0, 1]
	y = np.asarray(y).reshape(4, 1)
	AI.train(2, y)

if __name__=='__main__':
	main()