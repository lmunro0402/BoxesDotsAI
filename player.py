# Player class
#
# Author: Luke Munro

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

 	def reset(self):
 		self.score = 0