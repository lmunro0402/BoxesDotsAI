# Minimax algorithm 
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
	def __init__(self, dim, base):
		Player.__init__(self, "Minimax " + str(base))
		self.dim = dim
		self.base = base


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
					score = max_play(clone, depth)
				else:
					score = min_play(clone, depth)
				branches[i] += score
		if DEBUG:
			G.display_game()
			print branches
			print moves
		if depth < 3: # RANDOM BAD AI
			rand_move = random.randint(0, num_best_moves(branches)-1)
		else: # TRAIN AI MAKE FIRST GOOD MOVE
			rand_move = 0
		return formatMoves(orderMoves(branches), moves)[rand_move]


	def check_ending_chain(self, game_state, DEBUG=False):
		G = Clone(self.dim, game_state)
		moves = G.find_moves()
		branches = [False] * len(moves)
		winning_move = moves[0]
		for i, move in enumerate(moves):
			clone = copy.deepcopy(G)
			old_usedBoxes = clone.usedBoxes
			clone.move(move)
			if old_usedBoxes < clone.usedBoxes:
				branches[i] = continue_chain(clone)
			else:
				branches[i] = False
		if DEBUG:
			G.display_game()
			print branches
			print moves
		# return filter(lambda x: x in branches, moves)
		return [moves[i] for i, branch in enumerate(branches) if branch]


def min_play(node, depth):
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
			score = min_play(clone, depth)
		else:
			score = max_play(clone, depth)
		if score < best_score:
			best_move = move
			best_score = score
	return best_score


def max_play(node, depth):
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
			score = max_play(clone, depth)
		else:
			score = min_play(clone, depth)
		if score >  best_score:
			best_move = move
			best_score = score
	return best_score


def continue_chain(node):
	if node.is_game_over():
		return True
	moves = node.find_moves()
	is_ending = False
	for move in moves:
		clone = copy.deepcopy(node)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		if old_usedBoxes < clone.usedBoxes:
			is_ending = continue_chain(clone)
	return is_ending


def main(): # FOR SCENARIO DEPTH TESTING
	m_state = [[1, 1, 1], [1, 0, 1, 1], [0, 0, 1], [1, 1, 0, 1], [0, 0, 0], [1, 1, 1, 1],\
	[1, 1, 1]]
	AI = Minimax(2, 0)
	print AI.check_ending_chain(m_state, True)


if __name__=="__main__":
	main()


