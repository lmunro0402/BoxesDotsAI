# Dots & Boxes
#
# Author: Luke Munro

import DeepNN as NN
import time, random
import numpy as np
from Player import Player
import Trainer
from Clone import *
from Minimax import *
import copy
import utils as UTIL

def main():
	dim = int(input("Size of grid: "))
	train = int(input("How many games: "))
	mode = int(input("Player 1 is?\n You (0) | Minimax (1): "))
	numMoves = 2*(dim**2+dim)
	if mode == 1:
		base = raw_input("Minimax bonus depth: ")
		player1 = Minimax(dim, base)
	elif mode == 0:
		name = "Human" #raw_input("Enter name: ")
		player1 = Player(name)
	mode2 = int(input("Who are you playing?\n ShallowBlue AI (0) | Minimax (1): "))
	if mode2 == 1:
		base2 = raw_input("Minimax bonus depth: ")
		player2 = AI = Minimax(dim, base2)
	elif mode2 == 0:	
		val = input("Load AI (0) | Initialize AI (1): ")
		if val == 1:
			weight_params = []
			for i in range(input("Input # of layers: ")):
				weight_params.append(input("Layer {0}\n Input # of nodes: ".format(i)))
			weight_params.append(numMoves) 
			print "Initializing layers - " + str(weight_params[:len(weight_params)-1])
			player2 = AI = NN.NNet(numMoves, weight_params, dim)
		elif val == 0:
			try:
				weight_params = map(int, np.loadtxt('weight_params.txt').tolist())
			except:
				print "Failed to load AI"
			print "Initializing layers - " + str(weight_params[:len(weight_params)-1])
			player2 = AI = NN.NNet(numMoves, weight_params, dim)
			AI.loadWeights()

	file_num = str(raw_input("Write data to which file? (#Depth1Depth2): "))

	if mode == 0:
		Trainer1 = Trainer.Trainer(AI, numMoves, dim,  player2.getName())
		g = Grid(dim, [player1, player2])
		for i in range(train):
			turns = random.randint(0,1) 
			print g.players[turns].getName() + " starts\n"
			g.display_game()
			while g.game_status():
				cPlayer = g.players[turns%2]
				check = cPlayer.getScore()
				print cPlayer.getName() + " your move"
				g.turn(cPlayer)
				if cPlayer.getName() != "Shallow Blue":
					Trainer1.record(g.old_state, g.game_state)
				print cPlayer.getName() + " move - " + str(cPlayer.last_move)
				g.display_game()
				print cPlayer.getName() + " your score is " + str(cPlayer.getScore()) + "\n"
				print "---- Next Move ----"
				if check == cPlayer.getScore():
				 	turns += 1
			g.show_results()
			if player1.getScore() > AI.getScore():
				print "RECORDED IN \"move_record3#H\""
				Trainer1.write_record(file_num)
			elif AI.getScore() > player1.getScore():
				print "Nice try, Minimax. Starting victory lap..."
				time.sleep(1)
				UTIL.relive_game(dim, Trainer1.pokedex)
				print "Victory lap finished."
				Trainer1.write_record("SB-Conquests")
			g.reset()
	else:
		Trainer1 = Trainer.Trainer(AI, numMoves, dim,  "Minimax - " + str(base))
		Trainer2 = Trainer.Trainer(AI, numMoves, dim, "Minimax - " + str(base2))
		g = Grid(dim, [player1, player2])
		for i in range(train):
			print "Game - " + str(i)
			turns = random.randint(0,1) 
			while g.game_status():
				cPlayer = g.players[turns%2]
				check = cPlayer.getScore()
				g.turn(cPlayer)
				# print cPlayer.getName() + " - " + str(cPlayer.base)
				# g.display_game()
				new_state = copy.deepcopy(g.game_state) # BREAKING CONNECTION
				if cPlayer == player1:
					Trainer1.record(g.old_state, new_state)
				else:
					Trainer2.record(g.old_state, new_state)
				# Trainer1.train_by_play(0.1, g.old_state, new_state)
				if check == cPlayer.getScore():
				 	turns += 1
			if mode2 == 1 and mode == 1:
				if list(base)[:3] == ["S", "E", "T"]:
					if player2.getScore() > player1.getScore():
						Trainer2.write_record(file_num)
				elif player1.getScore() > player2.getScore():
					Trainer1.write_record(file_num)
				elif player2.getScore() > player1.getScore():
					Trainer2.write_record(file_num)
				else:
					print "Whatever you did. Stop doing it."
			elif player1.getScore() > AI.getScore():
				Trainer1.write_record("MINvsSB")
				print "Ehh lucky."
			elif AI.getScore() > player1.getScore():
				print "Nice try, Minimax. Starting victory lap..."
				time.sleep(1)
				UTIL.relive_game(dim, Trainer1.pokedex)
				print "Victory lap finished."
			Trainer1.clear_record()
			Trainer2.clear_record()
			g.show_results()
			g.reset()
	print "DONE PLAYING"
	if player2.getName() == "ShallowBlue":
		choice = input("Save weights?\n No (0) | Yes (1) (Overides existing!): ")
		if choice == 1:
			AI.writeWeights()
			print "Weights saved."
			np.savetxt('weight_params.txt', np.asarray(weight_params))
	print "Exiting..."




class Grid:
	def __init__(self, dim, players):
		""" Only square games allowed"""
		self.dim = dim 
		assert self.dim < 10, "Less than 10 please" 
		self.usedBoxes = 0
		self.players = players
		self.game_state = []
		for i in range(self.dim):
			self.game_state.append([0]*dim)
			self.game_state.append([0]*(dim+1))
		self.game_state.append([0]*dim)
		self.old_state = []


	def reset(self):
		self.players[0].reset()
		self.players[1].reset()
		self. usedBoxes = 0
		for i, row in enumerate(self.game_state):
			for x in range(len(row)):
				self.game_state[i][x] = 0


# ------------------ Funcs for minimax -----------------

	def get_depth(self, base): # IMPROVE THIS WITH THINKING
		moves_made = sum(UTIL.clean_game_state(self.game_state))
		num_moves = 2*(self.dim**2+self.dim)
		available_moves = num_moves - moves_made
		if list(base)[:3] == ["S", "E", "T"]: # this can be better!!!!! WAY BETTER!!!
				depth = int(list(base)[3])
		elif available_moves > self.dim*(self.dim+1)-2:
			depth = 2
		else:
			depth = int(base)+2
		return depth



	def getDim(self):
		return self.dim

	def add_players(self, players):
		self.players = players

	def turn(self, player):
		self.old_state = copy.deepcopy(self.game_state) # BREAKING CONNECTION
		if player.getName() == "almost Minimax":
			move = player.getMove(self.game_state, self.get_depth(player.base)).split(" ")
		else:
			move = player.getMove(self.game_state).split(" ")
		while not self.valid_move(move):
			print 'Invalid Move'
			move = player.getMove(self.game_state).split(" ")
		move = [int(x) for x in move]
		player.last_move = move
		self.move(move[0], move[1])
		self.update_scores(player)

	def move(self, row, index):
		self.game_state[row][index] = 1

#------------------------------ Checks ----------------------------------

	def valid_move(self, move): 
		try:
			move = [int(x) for x in move]
			row = move[0]
			index = move[1]
			if (row%2 == 0 and index > self.dim-1) or\
		 	(row%2 == 1 and index > self.dim) or (row > self.dim*2): 
				return False
			elif self.game_state[row][index] == 1:
				return False
			return True
		except:
			return False

# ---------------------------- Scoring -----------------------------------

	def game_status(self):
		return (self.dim**2) != self.usedBoxes

	def update_scores(self, player): 
		count = sum(self.check_boxes())
		if count != self.usedBoxes:
			diff = abs(self.usedBoxes-count)
			if diff == 1:
				player.plusOne()
			else:
				player.plusOne()
				player.plusOne()
			self.usedBoxes = count


	def get_boxes(self):
		"'Converts game_state into list of each box, contains repeats.'"
		boxes = []
		box_scores = []
		for i in range(0, self.dim*2, 2):
			# Go by rows
			for j in range(self.dim):
			# Each box
				boxes.append([self.game_state[i][j], self.game_state[i+1][j], \
				self.game_state[i+1][j+1], self.game_state[i+2][j]])
		return boxes

	def check_boxes(self):
		boxes = self.get_boxes()
		box_scores = [sum(x)//4 for x in boxes]
		return box_scores

# ------------------------- Display methods --------------------------------

	def display_moves(self):
		return self.game_state

	def display_game(self):
		buffer = [] #what is this
		hLine = "+---"
		hEmpty = "+   "
		vLine = "|   "
		vEmpty = "    "
		# Top row
		for i in range(self.dim):
			if self.game_state[0][i] == 1:
				buffer.append(hLine)
			else: buffer.append(hEmpty)
		buffer.append("+\n")
		# Middle rows
		for i in range(1, self.dim*2, 2):
			# Make horizontal passes
			for j in range(self.dim+1):
				if self.game_state[i][j] ==  1:
					buffer.append(vLine)
				else: buffer.append(vEmpty)
			buffer.append("\n")
			# Vertical passes
			for j in range(self.dim):
				if self.game_state[i+1][j] == 1:
					buffer.append(hLine)
				else: buffer.append(hEmpty)
			buffer.append("+\n")
		print "".join(buffer)

	def show_results(self):
		# print "---GAME RESULTS---"
		print self.players[0].getName() + " score is " + str(self.players[0].getScore())
		print self.players[1].getName() + " score is " + str(self.players[1].getScore())
		if self.players[0].getScore() == self.players[1].getScore():
			print "Tie"
		elif self.players[0].getScore() > self.players[1].getScore():
			print "Winner is " + self.players[0].getName()
		else:
			print "Winner is " + self.players[1].getName()

# -------------------------- Data methods for AI ---------------------------
	
	def get_data(self):
		moves = [i for x in self.game_state for i in x]
		# scores = [players[0].getScore(), players[1].getScore()]
		return str(moves) # + scores)
	def train_data(self):
		return [i for x in self.game_state for i in x]

if __name__ == "__main__":
	main()
