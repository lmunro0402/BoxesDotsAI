# Boxes and Dots Game
#
# Author: Luke Munro

import DeepNN as NN
import BoxesDots as BD
from Player import Player
from Trainer import Trainer
from Minimax import Minimax
import utils as UTIL
import sys as SYS

import time, random
import numpy as np
import copy


def main():
	try:
		dim = int(SYS.argv[1])
		num_games = int(SYS.argv[2])
		mode = int(SYS.argv[3])
		if mode:
			base = SYS.argv[4]
		mode2 = int(SYS.argv[5])
		if mode2:
			base2 = SYS.argv[6]
		if mode and mode2:
				file_num = SYS.argv[7]
		DISPLAY = False
		TRAINING = True
	except:
		dim = input("Size of grid: ")
		num_games = input("How many games: ")
		mode = input("Player 1 is?\n You (0) | Minimax (1): ")
		if mode:
			base = raw_input("Minimax bonus depth: ")
		mode2 = input("Who are you playing?\n ShallowBlue AI (0) | Minimax (1): ")
		if mode2:
			base2 = raw_input("Minimax bonus depth: ")
		if mode and mode2:
			file_num = raw_input("Write data to which file? (#Depth1Depth2): ")
		DISPLAY = True
		TRAINING = False
	
	numMoves = 2*(dim**2+dim)

	if mode == 1:
		player1 = Minimax(dim, base, True)
	elif mode == 0:
		player1 = Player("Human")
	else:
		print "Unknown command."
	if mode2 == 1:
		player2 = AI = Minimax(dim, base2, False)
	elif mode2 == 0:	
		player2 = AI = NN.NNet(numMoves, dim)
		try:
			weight_params = map(int, np.loadtxt('weight_params.txt').tolist())
			print "Loaded layers - " + str(weight_params[:len(weight_params)-1])
		except:
			print "Run Trainer.py to create ShallowBlue."
			raise SystemExit
	else:
		print "Unknown command."

	Trainer1 = Trainer(numMoves, dim, player1)
	Trainer2 = Trainer(numMoves, dim, player2)

	G = BD.BoxesDots(dim, [player1, player2])
	for i in range(num_games):
		turns = random.randint(0,1) 
		if DISPLAY == True:
			print G.players[turns].getName() + " starts\n"
			G.display_game()
		# GAME LOOP
		while G.game_status():
			cPlayer = G.players[turns%2]
			check = cPlayer.getScore()
			G.turn(cPlayer)
			new_state = copy.deepcopy(G.game_state) # BREAKING CONNECTION
			if cPlayer == player1:
				Trainer1.record(G.old_state, new_state)
			else:
				Trainer2.record(G.old_state, new_state)
			if DISPLAY == True:
				print cPlayer.getName() + " your move"
				print cPlayer.getName() + " move - " + str(cPlayer.last_move)
				G.display_game()
				print cPlayer.getName() + " your score is " + str(cPlayer.getScore()) + "\n"
				print "---- Next Move ----"
			if check == cPlayer.getScore():
			 	turns += 1
		G.show_results()

		if i%(round(num_games/3.0)+1) == 0 and TRAINING: # ARTIFICIAL DATA IS BEING CREATED
			progress = str(round((float(i)/num_games)*100)) + "% completed " + file_num + ".\n"
			with open('{0}_progress.txt'.format(file_num), 'a') as f:
				f.write(progress)
			UTIL.send_mail(progress)
			time.sleep(5)

		if mode and mode2:
			if list(base)[:3] == ["S", "E", "T"]:
				if player2.getScore() > player1.getScore():
					Trainer2.write_record(file_num)
					print "Game recorded in \"move_record3#" + str(file_num) + "\""
				else:
					print "Game not logged"
			elif player1.getScore() > player2.getScore():
				print "Game recorded in \"move_record3#" + str(file_num) + "\""
				Trainer1.write_record(file_num)
			else:
				print "Game recorded in \"move_record3#" + str(file_num) + "\""
				Trainer2.write_record(file_num)
		elif not mode:
			if player1.getScore() > AI.getScore():
				print "Ehh. lucky"
				print "Game recorded in \"move_record3#H\""
				Trainer1.write_record("Human-Conquests")
			else:
				print "Nice try, Human."
				print "Game logged in \"move_record3#AI-Conquests.txt\""
				Trainer2.write_record("AI-Conquests")
		else:
			if player1.getScore() > AI.getScore():
				print "Game recorded in \"move_record3#SB-L-MM.txt\""
				Trainer1.write_record("SB-L-MM")
			else:
				print "HE DID IT!!"
				print "Game recorded in \"move_record3#SB-W-MM.txt\""
				Trainer2.write_record("SB-W-MM")
		Trainer1.clear_record()
		Trainer2.clear_record()
		G.reset()
	if TRAINING:
		UTIL.send_mail("Finished " + file_num + ".")
	print "Done playing."
	print "Exiting..."


if __name__ == "__main__":
	main()
