# Training Object
#
# Author: Luke Munro

import numpy as np 

class Trainer:
	def __init__(self, sizeIn, gridSize):
		self.gridSize = gridSize
		self.sizeIn = sizeIn


	def format_data(self, state):
		return [int(i) for x in state for i in x]


	def record(self, old_state, new_state):
		new_state = self.format_data(new_state)
		with open("move_record.txt", "a") as record:
			record.write(str(old_state)+'\n')
			record.write(str(new_state)+'\n')
			record.write("#---------- Next Move ------------\n")

	def train_by_player(self, old_state, new_state):
		return None

	def train_from_record(self):
		return None