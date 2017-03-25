# Reformat data into CSV
#
# @autor Luke Munro

import Trainer, csv
import DeepNN as NN
import sys as SYS

Ash = Trainer.Trainer(24, 3, NN.NNet(24, 3))
f = open("move_record3#{0}.csv".format(SYS.argv[1]), "wt")
writer = csv.writer(f)
raw_data = Ash.data_from_record(SYS.argv[1])
for pair in raw_data:
	old_state = pair[0]
	new_state = pair[1]
	move = Ash.get_training_move(old_state, new_state).reshape(1, 24).tolist()[0]
	old_state.append(move.index(1))
	print move
	writer.writerow(old_state)
f.close()