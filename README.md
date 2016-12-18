# Deep Learing Network for Boxes&Dots 
###Framework for a deep learning network to play Boxes &amp; Dots. NN Framework is easily adaptable.
##Objective: Create an AI capable of mimicing the minmax algorithm. Experiment with layers and nodes needed. Beat people in a 3 x 3 Boxes and Dots game.

##Strategy: 
- 1) Start with 2 x 2 board
- 2) Play people 
    - Supervised learning from people's moves
    - Log all lost games
- 2) Create minmax algorithm
    - Experiment with # of layers and nodes 
    - Have DLN replicate minmax algorithm
      

##Current Progress:
- Boxes & Dots game created                                                 11/24
- Basic AI w/ single hidden layer                                           11/26
- Attempt to train AI through mutation                                      11/27
- Backpropagation working                                                   11/28
- Framework for Deep Learning AI                                            12/5
- Training data storage, interalized data transfer                          12/6
- Trainer created, training for data working, started collecting            12/8
- Implemented gradient descent w/ momentum 				    12/9
- Implemented Nesterov accelerated gradient                                 12/10
- Started work on minimax                                                   12/16
- Created beta minimax w/ depth                                             12/17


#Goals & Thoughts:
- Optimization of Gradient Descent by decreasing alpha on cost function and/or theta bounces
	- Start with first layer deeper layers weights not stable 
	- Momentum and Nesterov accelerated gradient
		See: http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
	
- Reinforcement Learning. 

- Minimax will pick brach with most winning futures which is not necessarily the branch with the highest probably winning outcome. Will attempt to get DLNN to surpass this minimax shortcoming. 
