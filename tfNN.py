##
# Neural Net w/ TensorFlow
#
# @author Luke Munro
##

import sys
import tflearn
import Trainer 
import utils as UTIL
import DeepNN as NN
from tflearn.data_utils import load_csv

file_num = sys.argv[1]
numMoves = 24
AI = NN.NNet(numMoves, 3, [10, numMoves])
trainer = Trainer.Trainer(24, 3, AI)

reader = tf.TextLineReader() # FIGURE THIS OUT, HOW TO SPLIT X AND Y
all_data = trainer.data_from_record(file_num)
data, labels = tflearn.load_csv(all_data)
data = []
labels = []
for pair in all_data:
	data.append(pair[0])
	labels.append(trainer.get_training_move(pair[0], pair[1]))


net = tflearn.input_data(shape=[None, 24])
net = tflearn.fully_connected(net, 100)
net = tflearn.fully_connected(net, 24, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=1, batch_size=1, show_metric=True)





# state_pair = [UTIL.assemble_state(dim, games[state_index][0]),\
# 		 								UTIL.assemble_state(dim, games[state_index][1])]
# UTIL.relive_game_from_file(dim, state_pair)