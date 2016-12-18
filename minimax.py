# Minimax algorithm 
#
# Author: Luke Munro

from simulator import *
import copy
from utils import orderMoves
from utils import formatMoves
from utils import makeCommands

# class mm_AI(Player):
# 	def __init__(self, sizeIn, gridSize):
# 		Player.__init__(self, "mmAI")

# def minimax(game_state):
# 	moves = 
# 	return max(
# 		)


def min_play(node, depth=5):
	node.depth += 1
	print node.depth
	if not node.is_game_over() or node.depth == depth:
		print "MIN_PLAY GAME OVER! CURRENT DEPTH - " + str(node.depth)
		return -node.score
	moves = node.find_moves()
	total_score = 0
	for move in moves:
		print str(moves) + " - " + str(move) + " - Min play"
		clone = copy.deepcopy(node)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		clone.display_game()
		if old_usedBoxes < clone.usedBoxes:
			clone.plus_one()
			score = min_play(clone, depth)
			# print score
		else:
			score = max_play(clone, depth)
		total_score += score
	return total_score
	# 	if score < best_score:
	# 		best_move = move
	# 		best_score = score
	# return best_score

def max_play(node, depth=5):
	node.depth += 1
	print node.depth
	if not node.is_game_over() or node.depth == depth:
		print "MAX_PLAY GAME OVER CURRENT DEPTH - " + str(node.depth)
		print "Depth - " + str(node.depth)
		return node.score
	moves = node.find_moves()
	total_score = 0
	for move in moves:
		# print "\n BRANCH BITCH"
		print str(moves) + " - " + str(move) + " - Max play" 
		clone = copy.deepcopy(node)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		clone.display_game()
		if old_usedBoxes < clone.usedBoxes:
			clone.plus_one()
			score = max_play(clone, depth)
			# print score
		else:
			score = min_play(clone, depth)
		total_score += score
	return total_score
	# 	score = min_play(clone)
	# 	if score < best_score:
	# 		best_score = move
	# 		best_score = score
	# return best_score


def minimax(dim, game_state, depth):
	G = Node(dim, game_state) # depth)
	moves = G.find_moves()
	branches = [0] * len(moves)
	G.display_game()
	for i, move in enumerate(moves):
	# for i, move in enumerate(moves):
		print "FIRST MOVE - " +  str(moves), move
		clone = copy.deepcopy(G)
		old_usedBoxes = clone.usedBoxes
		clone.move(move)
		clone.display_game()
		if old_usedBoxes < clone.usedBoxes:
			branches[i] += 1
			score = max_play(clone, 2)
		else:
			score = min_play(clone, 2)
		branches[i] += score
	G.display_game()
	print moves
	print branches
	return formatMoves(orderMoves(branches), moves)



def main():
	m_state = [[1, 1], [1, 1, 1], [1, 1], [1, 1, 0], [0, 0]]
	print minimax(2, m_state, 0)


if __name__=="__main__":
	main()


