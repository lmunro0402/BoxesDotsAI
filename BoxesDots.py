# Dots & Boxes
#
# Author: Luke Munro

import DeepNN as NN
import time, random
import numpy as np
from player import Player
import Trainer

def main():
	dim = int(input("Size of grid: "))
	train = int(input("How many games: "))
	numMoves = 2*(dim**2+dim)
	player1 = AI = NN.NNet(numMoves, [50, 30, numMoves], dim) 
	name = raw_input("Enter name: ")
	player2 = Player(name)
	val = input("1 for load weights 0 for no: ")
	if  val == 1:
		AI.loadWeights()

	ProfOak = Trainer.Trainer(AI, numMoves, dim,  player2.getName())

	g = Grid(dim, [player1, player2])

	for i in range(train):
		turns = random.randint(0,1) * 0
		print g.players[turns].getName() + " starts\n"
		g.display_game()
		while g.game_status():
			cPlayer = g.players[turns%2]
			check = cPlayer.getScore()
			print cPlayer.getName() + " your move"
			g.turn(cPlayer)
			if cPlayer.getName() != "AI":
				ProfOak.record(g.old_state, g.game_state)
				ProfOak.train_by_play(0.1, g.old_state, g.game_state)
			print cPlayer.getName() + " move - " + str(cPlayer.last_move)
			g.display_game()
			print cPlayer.getName() + " your score is " + str(cPlayer.getScore()) + "\n"
			print "---- Next Move ----"
			if check == cPlayer.getScore():
			 	turns += 1
		g.show_results()
		if AI.getScore() < player2.getScore():
			ProfOak.write_record()
		g.reset()
	AI.writeWeights()




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


	def getDim(self):
		return self.dim

	def add_players(self, players):
		self.players = players

	def turn(self, player):
		self.old_state = [int(i) for x in self.game_state for i in x]
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

# Figure out errors for this - works now but ehh
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

	def update_scores(self, player): # THIS IS BUGGED sometimes need +2 : FIXED
		count = sum(self.check_boxes())
	#	print count
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
		buffer = [] #buffer? what is this
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
		print "---GAME RESULTS---"
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
