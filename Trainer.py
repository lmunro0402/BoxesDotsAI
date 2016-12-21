# Training Object
#
# Author: Luke Munro

import numpy as np 
import DeepNN as NN
import utils as UTIL

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
		self.pokedex.append([old_state, game_state])


	def write_record(self, file_num):
		if file_num == "NA":
			pass
		else:
			with open("move_record{0}#{1}.txt".format(self.gridSize, file_num), "a") as record:
				record.write("#------- Player = " + self.pokedex[0] + " ------------- \n")
				for pair in self.pokedex[1:]:
					old_state = self.format_game_state(pair[0])
					new_state = self.format_game_state(pair[1]) 
					record.write(str(old_state)+"\n")
					record.write(str(new_state)+"\n")
					record.write("#---------- Next Move ------------\n")
				record.write("#\n")
			# CLEARING RECORD
			self.clear_record()

	def clear_record(self):
		self.pokedex = [self.playerName]


	def data_from_record(self, file_num):
		data = []
		with open("move_record{0}#{1}.txt".format(self.gridSize, file_num), 'r') as record:
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

	def get_training_move(self, old_state, new_state):
		size = len(old_state)
		new_state = np.asarray(new_state).reshape(size, 1)
		old_state = np.asarray(old_state).reshape(size, 1) 
		move = new_state - old_state
		return move

	def train_by_play(self, alpha, old_state, game_state):
		new_state = self.format_game_state(game_state)
		old_state = self.format_game_state(old_state)
		self.train_AI(alpha, old_state, new_state)


	def train_from_record(self, alpha, file_num):
		training_data = self.data_from_record(file_num)
		print len(training_data)
		for i, pair in enumerate(training_data):
			old_state = pair[0]
			new_state = pair[1]
			self.train_AI(alpha, old_state, new_state)
			if i%1000 == 0:
				print i
				self.AI.writeWeights()


def main():
	dim = int(input("Game size: "))
	numMoves = 2*(dim**2+dim)
	mode1 = input("Train (0) | View recorded games (1): ")
	file_num = raw_input("Input file extension: ")
	AI = NN.NNet(numMoves, [50, 30, numMoves], dim)
	Ash = Trainer(AI, numMoves, dim)
	if mode1 == 1:
		print "Please wait..."
		print "FYI - These are one-sided, past state then new state."
		games = Ash.data_from_record(file_num)
		print "# Moves - " + str(len(games))
		for state_index in range(0, len(games)):
			# print games[state_index]
			state_pair = [UTIL.assemble_state(dim, games[state_index][0]),\
				 								UTIL.assemble_state(dim, games[state_index][1])]
			UTIL.relive_game_from_file(dim, state_pair)
			# print state_pair
			raw_input("Press Enter to continue:")
	elif mode1 == 0:
		mode2 = input("Load weights (0) | Random AI (1) (This WILL overwrite your old AI): ")
		if mode2 == 0:
			AI.loadWeights()
		print AI.getWeights()[0][0]
		print AI.getWeights()[1][0]
		alpha = input("Training Rate = ")
		Ash.train_from_record(alpha, file_num)
		print AI.getWeights()[0][0]
		print AI.getWeights()[1][0]

if __name__=="__main__":
	main()