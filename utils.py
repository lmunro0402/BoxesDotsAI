# Utilities for translating AI moves to game
#
# Author: Luke Munro

import numpy as np

# ---------- For Neural Net ----------------

def orderMoves(probs): # CONDENSE fix for same prob
	moves = []
	probs = np.array(probs).tolist()
	tProbs = list(probs)
	for i in range(len(probs)):
		high = max(tProbs)
		index = probs.index(high)
		probs[index] = None # working fix for same probs bug
		moves.append(index)
		tProbs.remove(high)
	return moves

# ----------- For minimax algorithm -----------

def find_moves(game_state, grid_dim):
	game_state = clean_game_state(game_state)
	moves = [x for x in range(len(game_state)) if game_state[x] == 0]
	return moves

def clean_game_state(game_state):
	return [x for row in game_state for x in row]


def makeCommands(grid_dim):
	moveCommands = []
	for i in range(grid_dim*2+1):
		if i%2==0:
			for x in range(grid_dim):
				moveCommands.append(str(i)+" "+str(x))
		else:
			for x in range(grid_dim+1):
				moveCommands.append(str(i)+" "+str(x))
	return moveCommands

def formatMoves(moveOrder, commands): # CONDENSE
	fmatMoves = []
	for move in moveOrder:
		fmatMoves.append(commands[move])
	return fmatMoves


def onlyLegal(moves, game_state): # CONDENSE
	legal_moves = []
	for i in range(len(game_state)):
		if game_state[i] == 0:
			legal_moves.append(i) 
	move_order = filter(lambda x: x in legal_moves, moves) #Remove illegal moves
	return move_order

def cleanData(raw):
	data = [[int(i)] for x in raw for i in x]
	data = np.array(data)
	return data

