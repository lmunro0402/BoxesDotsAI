# Utilities for translating AI moves to game
#
# Author: Luke Munro
import smtplib
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

def find_moves(game_state, grid_dim):
	game_state = clean_game_state(game_state)
	moves = [x for x in range(len(game_state)) if game_state[x] == 0]
	return moves


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

# ----------- For minimax algorithm -----------

def clean_game_state(game_state):
	return [x for row in game_state for x in row]

def num_best_moves(move_ratings):
	possible_best_moves = 0
	best_score = max(move_ratings)
	for score in move_ratings:
		if score == best_score:
			possible_best_moves += 1
	return possible_best_moves

# ---------------------------------------------

def relive_game(dim, game_record):
	for pair in game_record[1:]:
		for state in pair:
			print "- Next Move -"
			display_game(dim, state)

def relive_game_from_file(dim, game_record):
	for state in game_record:
		print "- Next Move -"
		display_game(dim, state)

def assemble_state(dim, game_record):
	game_state = []
	i = 0
	for x in range(dim*2+1):
		if x%2==0:
			game_state.append(game_record[i:i+3])
			i += 3
		else:
			game_state.append(game_record[i:i+4])
			i += 4
	return game_state



def display_game(dim, game_state):
	buffer = [] #what is this
	hLine = "+---"
	hEmpty = "+   "
	vLine = "|   "
	vEmpty = "    "
	# Top row
	for i in range(dim):
		if game_state[0][i] == 1:
			buffer.append(hLine)
		else: buffer.append(hEmpty)
	buffer.append("+\n")
	# Middle rows
	for i in range(1, dim*2, 2):
		# Make horizontal passes
		for j in range(dim+1):
			if game_state[i][j] ==  1:
				buffer.append(vLine)
			else: buffer.append(vEmpty)
		buffer.append("\n")
		# Vertical passes
		for j in range(dim):
			if game_state[i+1][j] == 1:
				buffer.append(hLine)
			else: buffer.append(hEmpty)
		buffer.append("+\n")
	print "".join(buffer)

# ---------------------------------------------------------------

def send_mail(msg):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login("tifmrp1324ip@gmail.com", "wussgood$$")
	server.sendmail("tifmrp1324ip@gmail.com", "6504006400@vtext.com", msg)
	server.quit() 
