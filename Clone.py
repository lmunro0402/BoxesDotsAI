# Game simulation for minimax 
#
# Author: Luke Munro

from utils import *
import copy
import random

class Clone:
	def __init__(self, dim, game_state=[]):
		self.score = 0
		self.depth = 0
		self.game_state = game_state
		if game_state == []:
			for i in range(self.dim):
				self.game_state.append([0]*dim)
				self.game_state.append([0]*(dim+1))
			self.game_state.append([0]*dim)
		self.dim = len(game_state)//2
		self.usedBoxes = 0
		self.check_boxes()

	def plus(self, points):
		self.score += points

	def is_game_over(self):
		return (self.dim**2) != self.usedBoxes

	def move(self, move):
		move = [int(x) for x in move.split(" ")]
		self.game_state[move[0]][move[1]] = 1
		self.check_boxes()



	def find_moves(self):
		game_state = clean_game_state(self.game_state)
		moves = [x for x in range(len(game_state)) if game_state[x] == 0]
		moves = formatMoves(moves, makeCommands(self.dim))
		return moves


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
		self.usedBoxes = sum(box_scores)


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
