# Neural Network
#
# Author: Luke Munro
from sigNeuron import *
from BoxesDots import *
import numpy as np


"""Neural net class for systems with any # of hidden layers."""

class Net(Player):
	def __init__(self, sizeIn, layerList, gridSize):
		Player.__init__(self, "AI")
		self.sizeIn = sizeIn
		self.gridSize = gridSize
		self.numLayers = len(layerList)
		self.layerList = layerList
		self.layers = [[] for x in range(self.numLayers)]
		for i in range(layerList[0]):
			self.layers[0].append(Neuron(self.sizeIn+1, i))
		for i, nodes in enumerate(layerList[1:]):
			for x in range(nodes):
				self.layers[i+1].append(Neuron(len(self.layers[i])+1, x))


	def getWeights(self): 
		layerWeights = []
		layerWeights.append(np.zeros(shape=(self.layerList[0], self.sizeIn+1)))
		for i in range(self.numLayers-1):
			layerWeights.append(np.zeros(shape=(self.layerList[i+1], self.layerList[i]+1)))
		for i, layer in enumerate(self.layers):
			for x, node in enumerate(layer):
				layerWeights[i][x] = node.getW()
		return layerWeights


	def writeWeights(self):
		layerWeights = self.getWeights()
		for i, layer in enumerate(layerWeights):
			np.savetxt('weight{0}.txt'.format(i), layer) # one file later


	def loadWeights(self): # BREAKS IF YOU ONLY HAVE 1 NODE IN A LAYER
		loadedWeights = []
		for i in range(self.numLayers):
			loadedWeights.append(np.loadtxt('weight{0}.txt'.format(i)))
		for i, layer in enumerate(self.layers):
			for x, node in enumerate(layer):
				node.assignW(loadedWeights[i][x])


	def updateWeights(self, newLayerWeights):
		for i, layer in enumerate(newLayerWeights):
			np.savetxt('weight{0}.txt'.format(i), layer)
		self.loadWeights()


	def getMove(self):
		a = []
		z = []
		a1 = getData()
		justMoves = a1
		a.append(addBias(a1))
		for i in range(self.numLayers):
			z.append(computeZ(self.layers[i], a[i]))
			temp = sigmoid(z[i])
			a.append(addBias(temp))
		# REMOVE BIAS IN OUTPUT
		out = np.delete(a[self.numLayers], 0, axis=0)
		print out
		moves = findMoves(out)
		print moves
		legalMoves = onlyLegal(moves, justMoves)
		print legalMoves
		nextMoves = formatMoves(legalMoves, makeCommands(self.gridSize))
		return [int(x) for x in nextMoves[0]]

# -------------------------------------------------------------------------------------------------------------------


	def train(self, alpha, y):
# ----- Leave steps split for easier comprehension ------
		a = []
		z = []
		a1 = getData()
		justMoves = a1 
		a.append(addBias(a1))
		for i in range(self.numLayers):
			z.append(computeZ(self.layers[i], a[i]))
			temp = sigmoid(z[i])
			a.append(addBias(temp))
		# print "-------------------------------------"
		# for i in z:
		# 	print i
		# REMOVE BIAS IN OUTPUT
		out = np.delete(a[self.numLayers], 0, axis=0)
		print out
		print 1./2 * sum(out - y)**2
		# print "-------------------------------------"
		# for i in a:
		# 	print i
		# print "-------------------------------------"
		# GET WEIGHTS WITHOUT BIAS 
		noBiasWeights = self.getWeights()
		# for i in noBiasWeights:
		# 	print i
		for i, weights in enumerate(noBiasWeights):
			noBiasWeights[i] = rmBias(weights)
		deltas = []
		# EDIT THIS IF CHANING COST FUNCTION
		initialDelta = (out - y) * sigGradient(z[len(z)-1])
		deltas.append(initialDelta)
		# print "------------------------------------"
		# REALLY CHECK THIS 
		for x in range(self.numLayers-2, -1, -1):
			# print x
			deltaIndex = (self.numLayers-2) - x
			# print deltaIndex
			delta = np.dot(noBiasWeights[x+1].transpose(), deltas[deltaIndex]) * sigGradient(z[x])
			deltas.append(delta)
		# print "----------------------------"
		# for i in deltas:
			# print i
		# print "----------------------------"
		Grads = []
		# REORDER DELTAS FROM FIRST LAYER TO LAST			
		for i, delta in enumerate(deltas[::-1]):
			# print delta
			# print a[i]
			Grads.append(delta*a[i].transpose())
		# print "----------------------------"
		# for i in Grads:
		# 	print i
		# print "-------------------------------"
		# print alpha * np.asarray(Grads)
		# print ""
		# print self.getWeights()
		# print "---------------------------------"
		updatedWeights = self.getWeights() + -alpha*np.asarray(Grads)
		# print updatedWeights
		# print "------------------------------------"
		# print self.getWeights()
		self.updateWeights(updatedWeights)
		# print "-------------------------------------"
		# print self.getWeights()



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




if __name__ == '__main__':
	main()

