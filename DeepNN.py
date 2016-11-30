# Neural Network
#
# Author: Luke Munro
from sigNeuron import *
from BoxesDots import *
import numpy as np


"""Neural net class for systems with ONE hidden layer, 
				eventually will work generically for more. """
class Net(Player):
	def __init__(self, nodes, gridSize):
		Player.__init__(self, "AI") # Net is istanceof Player object
# ------ One Hidden Layer ------
# Nodes = # nodes in layer | sizeX = # data points | ouptuts = # possible ouptuts
# ----------- All sizes do NOT CONTAIN BIAS UNITS -----------
		self.gridSize = gridSize
		self.sizeX = 2*(gridSize*(gridSize+1)) #+2 
# ------ Hidden layer variables -------------------
		self.hidden = []
		self.sizeH = nodes
#-------- Output variables ----------
		self.outs = []
		self.sizeO = self.sizeX 
		for i in range(nodes):
			self.hidden.append(Neuron(self.sizeX+1, i))
		for i in range(self.sizeO): 
			self.outs.append(Neuron(self.sizeH+1, i))  
		self.layers = [self.hidden, self.outs]


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
	y = np.zeros(shape=(12, 1))
	t = np.zeros(shape=(12, 1))+0.001
	
	#y = addBias(y)
	x= [[ 0.44841772], [ 0.44841772], [ 0.44841772], [ 0.44841772], [ 0.3939411 ], [ 0.3939411 ], [ 0.3939411 ], [ 0.3939411 ], [ 0.43916391], [ 0.43916391], [ 0.43916391], [ 0.43916391]]
	# print y, y.shape
	a4 = np.zeros(shape=(12, 1))
	for i in range(12):
		a4[i] = x[i]
	aI = Net(10, 2)
	aI.loadWeights()
	a = []
	z = []
	a1 = getData()
	justMoves = a1 #list(a0[:len(a0)-2])
	a1 = addBias(a1)
	print a1, a1.shape
	a.append(a1)
	z2 = computeZ(aI.layers[0], a1)
	z.append(z2)
	a2 = sigmoid(z2)
	a2 = addBias(a2)
	a.append(a2)
	z3 = computeZ(aI.layers[1], a2)
	z.append(z3)
	a3 = sigmoid(z3)
	a.append(a3)
	print "------- costs -------"
	print costLog(y, a3)
	# print costMeanSquared(y, a3)
	delta3log = a3 - y
	delta3squared = (a3 - y) * sigGradient(z3)
	print "------- deltas3s -------"
	print delta3log, delta3log.shape
	# print ''
	# print delta3squared
	w = aI.getWeights()
	print "----- regularization -----"
	print reg(w, 1)
	print "------- delta2s --------"
	print "weight dims - " + str(w[0].shape), str(w[1].shape)
	wNoBias = np.delete(w[1], 0, axis=1)
	delta2log = np.dot(wNoBias.transpose(), delta3log) * sigGradient(z2)
	delta2squared = np.dot(wNoBias.transpose(), delta3squared) * sigGradient(z2)
	print delta2log, delta2log.shape
	# print ""
	# print delta2squared
	w2Gradlog = delta3log * a2.transpose() 
	w2GradSquared = delta3squared * a2.transpose()
	print"-------- gradients weights 2 --------"
	print w2Gradlog, w2Gradlog.shape
	# print w2GradSquared
	print "------- gradient weights 1 --------"
	w1Gradlog = delta2log * a1.transpose()
	w1GradSquared = delta2squared * a1.transpose()
	print w1Gradlog, w1Gradlog.shape
	# print w1GradSquared 
	print "--------- weights ----------"
	print w[0]
	print ""
	print w[1]
	print "--------- added gradient ---------"
	w1 = w[0] + w1Gradlog
	w2 = w[1] + w2Gradlog
	print w1
	print ""
	print w2


	# print sigmoid(a)
	# print 1 - sigmoid(a)
	# grad = sigGradient(z)
	# delta3 = y - a # * a2
	# delta2 = delta3 * sigGradient(z2) # a1


if __name__ == '__main__':
	main()

