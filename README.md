# ShallowBlue - A Boxes&Dots AI
### ShallowBlue (SB) is a Deep Learning Neural Network (DLNN) built from scratch off of numpy. SB is trained using gradient descent (specifically Nesterov accelerated gradient) from recorded games. Recorded games are generated with the minimax algorithm and expert players.

##Objective: Create a competitve DLNN to play Boxes and Dots.

##Strategy: 
- 1) Start with 2 x 2 board
- 2) Play people 
    - Supervised learning from people's moves
    - Log all lost games
- 2) Create minmax algorithm
    - Experiment with # of layers and nodes 
    - Have DLN replicate minmax algorithm
- 3) Ply experts 
    - Make correct decisions at critcal states mid game
    - Play minimax w/ more depth

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
- Fixed minimax bugs, starting to train                                     12/18
- Minimax picks a random good move from best valued not first               12/19
- Experimenting with depth and when to increase depth w/ minimax            12/19
- Optimized minimax depth for recording games 				    12/20


#Goals & Thoughts:
- Optimization of Gradient Descent by decreasing alpha on cost function and/or theta bounces
	- Start with first layer deeper layers weights not stable 
	- Momentum and Nesterov accelerated gradient
		See: http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
	
- Reinforcement Learning. 

- UC Berkeley has noted several strategies other than those listed here. I will not be looking into other algorithms since the focus of this project is the DLNN and minimax is sufficient for 3 x 3 games. 
See: https://math.berkeley.edu/~berlek/cgt/dots.html

- Minimax Algorithm

- As this project is drawing to an end, I'm realizing what's really needed is CPU power and tons^(tons) of training data. Playing and training against the minimax works well but games take too long ~ 5 sec w/ current depth. To actually train Shallow Blue, I'll need in the millions of recorded games. I could optimize the Minimax algorithm more (pruning), or even implement the Monte Carlo search tree to speed things up. However, the real issue is CPU power. Even with the necessary data just trainng takes a very long time. So, get more CPU power, optimize minimax, or a lot of time to get Shallow Blue to a competitve level.

- Furthermore, some of the libraries I use run slowly. Specifically the deepcopy() method from copy is used frequently. 

- Remotely train on Raspberry Pi (will have to run for days)
