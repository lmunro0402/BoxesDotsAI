# Trainer Class
#
# Author: Luke Munro

import numpy as np 
import DeepNN as NN
import utils as UTIL
import sys as SYS
import time
from Minimax import Minimax

class Trainer:
	def __init__(self, sizeIn, gridSize, AI):
		self.gridSize = gridSize
		self.sizeIn = sizeIn
		self.AI = AI # can just be a player but we only train AIs
		self.playerName = AI.getName()
		self.pokedex = [self.playerName]
		self.Minimax = Minimax(3, 0)


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
				for pair in self.pokedex[5:]:
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


	def train_AI(self, alpha, old_state, new_state, OPTIMIZED):
		y = self.get_training_move(old_state, new_state)
		data_input = self.remake_games(3, old_state) + old_state
		if OPTIMIZED:
			self.AI.trainNAG(alpha, old_state, y, 0.4)
		else:
			self.AI.train(alpha, old_state, y)

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


	def train_from_record(self, alpha, file_num, OPTIMIZED=True):
		training_data = self.data_from_record(file_num)
		num_moves = len(training_data)
		print "Total moves - " + str(num_moves)
		print "Current progress: "
		for i, pair in enumerate(training_data):
			old_state = pair[0]
			new_state = pair[1]
			self.train_AI(alpha, old_state, new_state, OPTIMIZED)
			if i%(round(num_moves/3.0)+1) == 0:
				progress = str(round(float(i)/num_moves, 2)*100) + "% completed " + file_num + "\n"
				with open('{0}_progress.txt'.format(file_num), 'a') as f:
					f.write(progress)
				UTIL.send_mail(progress)
				time.sleep(5)
				self.AI.writeWeights()
		self.AI.writeWeights()

	def remake_games(self, dim, clean_game_state):
		game_state = UTIL.assemble_state(dim, clean_game_state)
		ranks = self.Minimax.rankMoves(game_state, 3)
		return ranks

def main():
	try:
		dim = int(SYS.argv[1])
		mode1 = int(SYS.argv[2])
	except:
		dim = int(input("Game size: "))
		mode1 = input("Train (0) | View recorded games (1) | Create new AI (2):  ")

	numMoves = 2*(dim**2+dim)

	if mode1 == 0 or mode1 == 1:
		try:
			file_num = SYS.argv[3]
		except:
			file_num = raw_input("Input extension of training file (string after #): ")
		if mode1 == 0:
			try:
				weight_params = map(int, np.loadtxt('weight_params.txt').tolist())
				print "Loaded layers - " + str(weight_params[:len(weight_params)-1])
				AI = NN.NNet(numMoves, dim)
			except:
				print "Failed to load AI. It seems somethings wrong. Try initilizing an AI."
				raise SystemExit		
			Ash = Trainer(numMoves, dim, AI)
			for layer in range(len(weight_params)):
				print AI.getWeights()[layer][0]
			print "Weight preview completed. "
			try:
				alpha = float(SYS.argv[4]) 
			except:
				alpha = input("Enter Training Rate = ")
			
			print "Extracting data. Please wait..."
			Ash.train_from_record(alpha, file_num)
			print "Trained weights preview"
			for layer in range(len(weight_params)):
				print AI.getWeights()[layer][0]
			final_msg = "Finished training."
			print final_msg
			UTIL.send_mail(final_msg)
		elif mode1 == 1:
			# CREATE A PLACEHOLDER AI FOR TRAINER OBJECT
			AI = NN.NNet(numMoves, dim, [10, numMoves])
			Ash = Trainer(numMoves, dim, AI)
			print "Please wait..."
			print "FYI - These are one-sided, past state then new state."
			games = Ash.data_from_record(file_num)
			print "# Moves - " + str(len(games))
			for state_index in range(0, len(games)):
				state_pair = [UTIL.assemble_state(dim, games[state_index][0]),\
					 								UTIL.assemble_state(dim, games[state_index][1])]
				UTIL.relive_game_from_file(dim, state_pair)
				raw_input("Press Enter to continue:")
	elif mode1 == 2:
		weight_params = []
		for i in range(input("Input # of layers: ")):
			weight_params.append(input("Layer {0}\n # of nodes: ".format(i)))
		weight_params.append(numMoves) 
		raw_input("Press enter to create new AI **WARNING: This overrides any existing AI! CTRL+C to exit now.**: ")
		print "Initializing layers - " + str(weight_params[:len(weight_params)-1])
		AI = NN.NNet(numMoves, dim, weight_params)
		np.savetxt('weight_params.txt', np.asarray(weight_params)) # using np cause it's shorter. 
		AI.writeWeights()
	else:
		print "Unsupported command."
	print "Done.\nExiting..."


if __name__=="__main__":
	main()