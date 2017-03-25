# TensorFlow for SB
#
# Luke Munro

import tensorflow as tf 

def main():

	x = tf.placeholder(tf.float32, [None, 24])
	W = tf.Variable(tf.zeros([24, 24]))
	b = tf.Variable(tf.zeros([24]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), redution_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	init = tf.globale_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	