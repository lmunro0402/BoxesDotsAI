# Training Object
#
# Author: Luke Munro

import numpy as np 
import DeepNN as NN

class Trainer:
	def __init__(self, AI, sizeIn, gridSize, playerName="Training"):
		self.gridSize = gridSize
		self.sizeIn = sizeIn
		self.AI = AI
		self.playerName = playerName
		self.pokedex = [playerName]


	def format_game_state(self, state):
		return [int(i) for x in state for i in x]


	def record(self, old_state, game_state):
		new_state = self.format_game_state(game_state)
		self.pokedex.append([old_state, new_state])


	def write_record(self):
		with open("move_record{0}.txt".format(self.gridSize), "a") as record:
			record.write("#------- Player = " + self.pokedex[0] + "------------- \n")
			for pair in self.pokedex[1:]: 
				record.write(str(pair[0])+"\n")
				record.write(str(pair[1])+"\n")
				record.write("#---------- Next Move ------------\n")
			record.write("#\n")
		# CLEARING RECORD
		self.pokedex = [self.playerName]


	def get_training_move(self, old_state, new_state):
		size = len(old_state)
		new_state = np.asarray(new_state).reshape(size, 1)
		old_state = np.asarray(old_state).reshape(size, 1) 
		move = new_state - old_state
		return move


	def data_from_record(self):
		data = []
		with open("move_record{0}.txt".format(self.gridSize), 'r') as record:
			for line in record.readlines():
				li = line.strip()
				if not li.startswith('#'):
					data.append(li)
		clean_data = []
		for state in data:
			clean_data.append([int(state[x]) for x in range(len(state)) if x%3==1])
		clean_data = [[clean_data[i], clean_data[i+1]] for i in range(0, len(clean_data)-1, 2)]
		return clean_data


	def train_AI(self, alpha, old_state, new_state):
		y = self.get_training_move(old_state, new_state)
		# self.AI.train(alpha, old_state, y)
		self.AI.trainNAG(alpha, old_state, y)


	def train_by_play(self, alpha, old_state, game_state): # FOR ON THE JOB TRAINING LOL ;)
		new_state = self.format_game_state(game_state)
		self.train_AI(alpha, old_state, new_state)


	def train_from_record(self, alpha):
		training_data = self.data_from_record()
		for i, pair in enumerate(training_data):
			print i
			old_state = pair[0]
			new_state = pair[1]
			self.train_AI(alpha, old_state, new_state)



def main():
	dim = int(input("Game size: "))
	numMoves = 2*(dim**2+dim)
	AI = NN.NNet(numMoves, [50, 30, numMoves], dim)
	ProfOak = Trainer(AI, numMoves, dim)
	AI.loadWeights()
	print AI.getWeights()[0][0]
	print AI.getWeights()[1][0]
	alpha = input("Training Rate = ")
	print alpha
	ProfOak.train_from_record(alpha)
	print AI.getWeights()[0][0]
	print AI.getWeights()[1][0]

if __name__=="__main__":
	main()