# Minimax Algorithm 
#
# Author: Luke Munro

from Clone import *
from Player import *
import copy
import random
from utils import orderMoves
from utils import formatMoves
from utils import makeCommands


class Minimax(Player):
	""" Mnimax algorithm as a player. """
	def __init__(self, dim, base, random=True):
		Player.__init__(self, "Minimax " + str(base))
		self.dim = dim
		self.base = base
		self.random = random

	def getMove(self, game_state, depth, DEBUG=False):
		G = Clone(self.dim, game_state) 
		moves = G.find_moves()
		branches = [0] * len(moves)
		best_score = -9e99
		best_move = moves[0] 
		# FIRST MOVE IS EVALUATED HERE FOR RANDOMNESS
		if depth > 0: 
			for i, move in enumerate(moves):
				clone = copy.deepcopy(G)
				old_usedBoxes = clone.usedBoxes
				clone.move(move)
				if old_usedBoxes < clone.usedBoxes:
					branches[i] += (clone.usedBoxes - old_usedBoxes)
					clone.depth += 1
					score = self.max_play(clone, depth)
				else:
					score = self.min_play(clone, depth)
				branches[i] += score
		if DEBUG:
			G.display_game()
			print branches
			print moves
		if self.random: # RANDOM BAD AI 
			rand_move = random.randint(0, num_best_moves(branches)-1)
		else: # TRAIN AI MAKE FIRST GOOD MOVE
			rand_move = 0
		return formatMoves(orderMoves(branches), moves)[rand_move]

	def min_play(self, node, depth):
		if node.is_game_over() or node.depth == depth:
			return node.score
		node.depth += 1
		moves = node.find_moves()
		best_score = 9e99
		for move in moves:
			clone = copy.deepcopy(node)
			old_usedBoxes = clone.usedBoxes
			clone.move(move)
			if old_usedBoxes < clone.usedBoxes:
				clone.plus(old_usedBoxes - clone.usedBoxes)
				score = self.min_play(clone, depth)
			else:
				score = self.max_play(clone, depth)
			if score < best_score:
				best_move = move
				best_score = score
		return best_score


	def max_play(self, node, depth):
		if node.is_game_over() or node.depth == depth:
			return node.score
		node.depth += 1
		moves = node.find_moves()
		best_score = -9e99
		for move in moves:
			clone = copy.deepcopy(node)
			old_usedBoxes = clone.usedBoxes
			clone.move(move)
			if old_usedBoxes < clone.usedBoxes:
				clone.plus(clone.usedBoxes - old_usedBoxes)
				score = self.max_play(clone, depth)
			else:
				score = self.min_play(clone, depth)
			if score >  best_score:
				best_move = move
				best_score = score
		return best_score

# -------------- ENDING SEQUENCE FOR SHALLOWBLUE -----------------


	def check_ending_chain(self, game_state, current_score, DEBUG=False):
		initial_state = Clone(self.dim, game_state)
		initial_state.plus(current_score)
		self.ENDING_SEQUENCE = self.continue_chain(initial_state)
		if DEBUG:
			G.display_game()


	def continue_chain(self, node, DEBUG=False):
		print node.score
		if node.is_game_over() or node.score >= 5:
			return True
		moves = node.find_moves()
		can_end = False
		for move in moves:
			if DEBUG:
				print move
				print can_end
			if can_end == True:
				break 
			clone = copy.deepcopy(node)
			old_usedBoxes = clone.usedBoxes
			clone.move(move)
			if old_usedBoxes < clone.usedBoxes:
				clone.plus(clone.usedBoxes - old_usedBoxes)
				can_end = self.continue_chain(clone)
		return can_end




def main(): # FOR SCENARIO DEPTH TESTING
	m_state = [[1, 1, 0], [1, 1, 1, 0], [0, 0, 0], [1, 1, 1, 0], [0, 0, 0], [1, 0, 1, 0],\
	[1, 1, 0]]
	AI = Minimax(2, 0)
	AI.check_ending_chain(m_state, True)
	print AI.ENDING_SEQUENCE


if __name__=="__main__":
	main()


