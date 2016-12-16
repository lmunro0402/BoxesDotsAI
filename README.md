# Deep Learing Network for Boxes&Dots 
###Framework for a deep learning network to play Boxes &amp; Dots. NN Framework is easily adaptable.
##Objective: Create an AI capable of mimicing the minmax algorithm. Experiment with layers and nodes needed. Beat people in a 3 x 3 Boxes and Dots game.

## Move Diagram

  +--0-0--+--0-1--+--0-2--+
  |       |       |       | 
 1|0     1|1     1|2     1|3 
  |       |       |       |
  +--2-0--+--2-1--+--2-2--+
  |       |       |       | 
 3|0     3|1     3|2     3|3 
  |       |       |       |
  +--4-0--+--4-1--+--4-2--+
  |       |       |       | 
 5|0     5|1     5|2     5|3 
  |       |       |       |
  +--6-0--+--6-1--+--6-2--+


##Strategy: 
- 1) Start with 2 x 2 board
- 2) Play people 
    - Supervised learning from people's moves
    - Log all lost games
- 2) Create minmax algorithm
    - Experiment with # of layers and nodes 
    - Have DLN replicate minmax algorithm
      

Current Progress:
- Boxes & Dots game created                                                 11/24
- Basic AI w/ single hidden layer                                           11/26
- Attempt to train AI through mutation                                      11/27
- Backpropagation working                                                   11/28
- Framework for Deep Learning AI                                            12/5
- Training data storage, interalized data transfer                          12/6
- Trainer created, training for data working, started collecting            12/8
- Implemented gradient descent w/ momentum 				    12/9
- Implemented Nesterov accelerated gradient                                 12/10


#Goals & Thoughts:
- Optimization of Gradient Descent by decreasing alpha on cost function and/or theta bounces
	- Start with first layer deeper layers weights not stable 
	- This is already a thing - Momentum and Nesterov accelerated gradient
		See: http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
	- Will attempt a modified version to improve one scenario. 
	- Drafted concept on paper. Unsure if actually applicable. See scan under Reflective momentum.
	- Not sure if these grad optimizations momentum, nesterov are the best for my situation since I am dealing with AI that has the play the entire game. Splitting my efforts with reinforcment learning.
	- Moving on to 3 x 3. Too easy to tie in 2 x 2.
	
- Reinforcement Learning. 
