import DeepNN as NN

def main():
	AI= NN.Net(3, [2, 3, 4])
	print AI.getWeights()

if __name__=='__main__':
	main()