# Dots & Boxes
#
# Author: Luke Munro

import DeepNN as NN
import time
import numpy as np
# -----------------------------------ADD MOVE DIAGRAM -----------------------

class Grid:
	def __init__(self, dim):
		""" Only square games allowed"""
		self.dim = dim 
		assert self.dim < 5, "Less than 5 please" # CHANGE COMMAND INPUT FOR BIGGER GAMES
		self.usedBoxes = 0
		self.players = []
		self.moves = []
		for i in range(self.dim):
			self.moves.append([0]*dim)
			self.moves.append([0]*(dim+1))
		self.moves.append([0]*dim)
		self.old_state = []

	def reset(self):
		self = Grid(self.dim)

	def getDim(self):
		return self.dim

	def add_players(self, players):
		self.players = players

	def turn(self, player, game_state):
		self.old_state = [int(i) for x in self.moves for i in x]
		move = player.getMove(game_state)
		while not self.valid_move(move):
			print 'Invalid Move'
			move = player.getMove(game_state)
		move = [int(x) for x in move]
		player.last_move = move
		self.move(move[0], move[1])
		self.update_scores(player)

	def move(self, row, index):
		self.moves[row][index] = 1

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
			elif self.moves[row][index] == 1:
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
		boxes = []
		box_scores = []
		for i in range(0, self.dim*2, 2):
			# Go by rows
			for j in range(self.dim):
			# Each box
				boxes.append([self.moves[i][j], self.moves[i+1][j], \
				self.moves[i+1][j+1], self.moves[i+2][j]])
		return boxes

	def check_boxes(self):
		boxes = self.get_boxes()
		box_scores = [sum(x)//4 for x in boxes]
		return box_scores

# ------------------------- Display methods --------------------------------

	def display_moves(self):
		return self.moves

	def display_game(self):
		buffer = [] #buffer? what is this
		hLine = "+---"
		hEmpty = "+   "
		vLine = "|   "
		vEmpty = "    "
		# Top row
		for i in range(self.dim):
			if self.moves[0][i] == 1:
				buffer.append(hLine)
			else: buffer.append(hEmpty)
		buffer.append("+\n")
		# Middle rows
		for i in range(1, self.dim*2, 2):
			# Make horizontal passes
			for j in range(self.dim+1):
				if self.moves[i][j] ==  1:
					buffer.append(vLine)
				else: buffer.append(vEmpty)
			buffer.append("\n")
			# Vertical passes
			for j in range(self.dim):
				if self.moves[i+1][j] == 1:
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
		moves = [i for x in self.moves for i in x]
		# scores = [players[0].getScore(), players[1].getScore()]
		return str(moves) # + scores)
	def train_data(self):
		return [i for x in self.moves for i in x]

# -------------------------- Player Class ----------------------------------
class Player:
	def __init__(self, name):
		self.name = name
		self.score = 0
		self.last_move = []

	def getName(self):
		return self.name

	def getMove(self, game_state):
		move = raw_input("Input 2 numbers: Row then Column (ex. first vertical line would be 10): ")
		return move

	def plusOne(self):
		self.score += 1

	def getScore(self):
 		return self.score

 # -------------------------------- Training ---------------------------------------------
def training_data(old_state, current_state):
	size = len(old_state)
	current_state = [int(i) for x in current_state for i in x]
	current_state = np.asarray(current_state).reshape(size, 1)
	old_state = np.asarray(old_state).reshape(size, 1) 
	move = current_state - old_state
	print move
	return move

def main():
	dim = int(input("Size of grid: "))
	train = int(input("How many games: "))
	numMoves = 2*(dim**2+dim)
	player1 = AI = NN.Net(numMoves, [30, 20, numMoves], dim) 
	name = raw_input("Enter name: ")
	player2 = Player(name)
	val = input("1 for load weights 0 for no: ")
	if  val == 1:
		AI.loadWeights()

	for i in range(train):
		g = Grid(dim)
		g.add_players([player1, player2])
		print g.players[0].getName() + " starts\n"
		turns = 0
		g.display_game()
		while g.game_status():
			cPlayer = g.players[turns%2]
			check = cPlayer.getScore()
			print cPlayer.getName() + " your move"
			g.turn(cPlayer, g.moves)
			if cPlayer.getName() != "AI":
				train_data = training_data(g.old_state, g.moves)
				AI.train(g.old_state, 0.1, train_data)
				# CREATE ALPHA OPTIMIZATION FUNCION COST BOUNCES
			print cPlayer.getName() + " move - " + str(cPlayer.last_move)
			g.display_game()
			print cPlayer.getName() + " your score is " + str(cPlayer.getScore()) + "\n"
			print "---- Next Move ----"
			if check == cPlayer.getScore():
			 	turns += 1
		g.show_results()
		g.reset()
	AI.writeWeights()

if __name__ == "__main__":
	main()
