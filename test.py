import DeepNN as NN

def main():
	AI= NN.Net(12, [2, 3, 12], 2)
	AI.loadWeights()
	print AI.getWeights()
	print " ----------------------------------- "
	print AI.getMove()

if __name__=='__main__':
	main()