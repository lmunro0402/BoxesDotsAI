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
	def __init__(self, dim):
		Player.__init__(self, "almost Minimax")
		self.dim = dim


	def getMove(self, game_state, depth):
		G = Clone(self.dim, game_state) # depth)
		# depth = 0
		# for row in G.game_state:
		# 	for link in row:
		# 		if link == 0:
		# 			depth += 1
		# depth = 2
		moves = G.find_moves()
		branches = [0] * len(moves)
		# G.display_game()
		best_score = -9e99
		best_move = moves[0]
		for i, move in enumerate(moves):
		# for i, move in enumerate(moves):
			# print "FIRST MOVE - " +  str(moves), move
			clone = copy.deepcopy(G)
			old_usedBoxes = clone.usedBoxes
			clone.move(move)
			# clone.display_game()
			if old_usedBoxes < clone.usedBoxes:
				branches[i] += 1
				score = max_play(clone, depth)
			else:
				score = min_play(clone, depth)
			branches[i] += score
			# if score > best_score:
			# 	best_move = move
			# 	best_score = score

			# break
		# G.display_game()
		# print moves
		# print branches
		# print formatMoves(orderMoves(branches), moves)
		# print num_best_moves(branches)
		rand_move = random.randint(0, num_best_moves(branches)-1)
		# print "Best score - " + str(best_score) + "Best move - " + str(best_move)
		return formatMoves(orderMoves(branches), moves)[rand_move]


def min_play(node, depth):
	node.depth += 1
	if not node.is_game_over() or node.depth == depth:
		# print "MIN_PLAY GAME OVER! CURRENT DEPTH - " + str(node.depth)
		return node.score
	moves = node.find_moves()
	best_score = 9e99
	for move in moves:
		# print str(moves) + " - " + str(move) + " - Min play"
		clone = copy.deepcopy(node)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		# clone.display_game()
		if old_usedBoxes < clone.usedBoxes:
			clone.plus(old_usedBoxes - clone.usedBoxes)
			score = min_play(clone, depth)
			# print score
		else:
			score = max_play(clone, depth)
		if score < best_score:
			best_move = move
			best_score = score
	return best_score


def max_play(node, depth):
	node.depth += 1
	if not node.is_game_over() or node.depth == depth:
		# print "MAX_PLAY GAME OVER CURRENT DEPTH - " + str(node.depth)
		return node.score
	moves = node.find_moves()
	best_score = -9e99
	for move in moves:
		# print str(moves) + " - " + str(move) + " - Max play" 
		clone = copy.deepcopy(node)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		# clone.display_game()
		if old_usedBoxes < clone.usedBoxes:
			clone.plus(clone.usedBoxes - old_usedBoxes)
			score = max_play(clone, depth)
			# print score
		else:
			score = min_play(clone, depth)
		if score >  best_score:
			best_move = move
			best_score = score
	return best_score


def main():
	m_state = [[1, 1], [0, 1, 0], [0, 0], [0, 1, 1], [1, 1]]
	AI = Minimax(2)
	print AI.getMove(m_state, 5)


if __name__=="__main__":
	main()


