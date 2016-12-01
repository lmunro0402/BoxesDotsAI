# Neural Network
#
# Author: Luke Munro
from sigNeuron import *
import numpy as np


"""Neural net class for systems with ONE hidden layer, 
				eventually will work generically for more. """
class Net():
	def __init__(self, sizeIn, layerList):
		self.sizeIn = sizeIn
		self.layers = [[] for x in range(len(layerList))]
		for i in range(sizeIn):
			self.layers[i].append(Neuron(self.sizeIn+1, i))
		for i in range(len(layerList)-1):
			self.layers[i+1].append(Neuron(len(self.layers[i])+1, i))
			print self.layers


	def getWeights(self): 
		layer1 = np.zeros(shape=(self.sizeH, self.sizeX+1))
		layer2 = np.zeros(shape=(self.sizeO, self.sizeH+1))
		for i, node in enumerate(self.hidden):
			layer1[i] = node.getW()
		for i, node in enumerate(self.outs):
			layer2[i] = node.getW()
		weights = [layer1, layer2]
		return weights

	def writeWeights(self):
		layerWeights = self.getWeights()
		layerWeights[0].tofile('weights1')
		layerWeights[1].tofile('weights2')


	def loadWeights(self):
		weights1 = np.fromfile('weights1').reshape(self.sizeH, self.sizeX+1)
		weights2 = np.fromfile('weights2').reshape(self.sizeO, self.sizeH+1)
		for i, weight in enumerate(weights1):
			self.hidden[i].assignW(weight)
		for i, weight in enumerate(weights2):
			self.outs[i].assignW(weight)

	def updateWeights(self, layerWeights):
		layerWeights[0].tofile('weights1')
		layerWeights[1].tofile('weights2')


	def getMove(self):
		a = []
		z = []
		a1 = getData()
		justMoves = a1 
		a.append(addBias(a1))
		z.append(computeZ(self.layers[0], a[0]))
		a.append(addBias(sigmoid(z[0])))
		z.append(computeZ(self.layers[1], a[1]))
		a.append(sigmoid(z[1]))
		moves = findMoves(a[2])
		legalMoves = onlyLegal(moves, justMoves)
		nextMoves = formatMoves(legalMoves, makeCommands(self.gridSize))
		return [int(x) for x in nextMoves[0]]


	def train(self, alpha, y):
# ----- Leave steps split for easier comprehension ------
		a = []
		z = []
		a1 = getData()
		justMoves = a1 
		a1 = addBias(a1)
		a.append(a1)
		z2 = computeZ(self.layers[0], a1)
		z.append(z2)
		a2 = sigmoid(z2)
		a2 = addBias(a2)
		a.append(a2)
		z3 = computeZ(self.layers[1], a2)
		z.append(z3)
		a3 = sigmoid(z3)
		a.append(a3)
		print costMeanSquared(y, a3)
		delta3 = (a3 - y)  * sigGradient(z3)
		layerWeights = self.getWeights()
		w1 = layerWeights[0]
		w2 = layerWeights[1]
		w1NoBias = rmBias(w1)
		delta2 = np.dot(w1NoBias, delta3) * sigGradient(z2)
		Grad1 = delta2 * a1.transpose()
		Grad2 = delta3 * a2.transpose()
		w1 += -alpha * Grad1
		w2 += -alpha * Grad2
		# Updating weights here
		for i, weights in enumerate(w1):
			self.hidden[i].assignW(weights)
		for i, weights in enumerate(w2):
			self.outs[i].assignW(weights)



# -------------------------- Computations -------------------------------

def computeZ(Nodes, X):
	w = np.zeros(shape=(np.size(Nodes), np.size(X)))
	for i, node in enumerate(Nodes):
		w[i] = node.getW()
	z = np.dot(w, X).reshape(np.size(Nodes), 1)
	return z

def sigmoid(z):
	return 1/(1+np.exp(-z))

def costLog(y, a):
	cost = -y * np.log10(a) - (1 - y) * np.log10(1 - a)
	return sum(cost)

def costMeanSquared(y, a):
	cost = ((a - y)**2)/2.0
	return sum(cost)

def sigGradient(z):
	return sigmoid(z) * (1 - sigmoid(z))

def reg(weights, Lambda): # Bias must be removed from weights
	w1= rmBias(weights[0])
	w2 = rmBias(weights[1])
	reg = Lambda/2.0 * (sum(sum(w1**2)) + sum(sum(w2**2)))
	return reg

def estimateGradlog(y, a, weights, epsilon): # DO THIS LATER
	for i in range(weights):
		for i in range(weights[i]):
			continue
	return None
	# return (costLog(y, a+epsilon) - costLog(y, a-epsilon))


# ---------------------------- Utility ----------------------------------

def getData():
	with open('data', 'r') as d:
		data = d.read()
	data = [[int(data[x])] for x in range(len(data)) if x%3==1]
	data = np.array(data)
	return data 
		
def addBias(aLayer): # Adds 1 to vertical vector matrix
	return np.insert(aLayer, 0, 1, axis=0)


def rmBias(weightMatrix): # removes bias weight from all nodes 
	return np.delete(weightMatrix, 0, axis=1)

# ------------------ Translating to BoxesDotes ----------------------


def findMoves(probs): # CONDENSE fix for same prob
	moves = []
	probs = probs.tolist()
	tProbs = list(probs)
	for i in range(len(probs)):
		high = max(tProbs)
		index = probs.index(high)
		probs[index] = -1 # working fix for same probs bug
		moves.append(index)
		tProbs.remove(high)
	return moves

def makeCommands(gridDim):
	moveCommands = []
	for i in range(gridDim*2+1):
		if i%2==0:
			for x in range(gridDim):
				moveCommands.append(str(i)+str(x))
		else:
			for x in range(gridDim+1):
				moveCommands.append(str(i)+str(x))
	return moveCommands

def formatMoves(moveOrder, commands): # CONDENSE
	fmatMoves = []
	for i, move in enumerate(moveOrder):
		fmatMoves.append(commands[move])
	return fmatMoves


def onlyLegal(moves, justMoves): # CONDENSE
	legalMoves = []
	for i in range(len(justMoves)):
		if justMoves[i] == 0:
			legalMoves.append(i)
	moveOrder = filter(lambda x: x in legalMoves, moves)
	return moveOrder



# --------------- for testing ----------------------------------
def main():
	AI = Net(3, [2, 3])

	# print sigmoid(a)
	# print 1 - sigmoid(a)
	# grad = sigGradient(z)
	# delta3 = y - a # * a2
	# delta2 = delta3 * sigGradient(z2) # a1


if __name__ == '__main__':
	main()

