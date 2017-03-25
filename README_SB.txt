Overview: ShallowBlue (SB) is an AI that utilizes a Neural Network (NN) and the Minimax algorithm to play Boxes and Dots. The NN is trained using stochastic gradient descent (Nesterov accelerated gradient) from data created by the Minimax algorithm playing itself with varying depths.


How does ShallowBlue work?

Early game - Establish usual board/grid: 
	While the grid is less than ~1/2 full, SB will take the first available move progressing from top left to bottom right. SB initializes a minimax of depth 2 is to take advantage of any mistakes made by the opponent (for an early box) and to prevent giving away any free boxes.

Mid game - Sacrifices and chain creation: 
	Mid game is general where the victor is decided. Here, a single move can finish a chain or split it in two smaller ones. Move order is very important. The Neural Network is actived to decide whether SB continue on it's trajectory or sacrifice boxes to win.

End game (kind of) - Look for ending sequence:
	 While the Neural Net plays a sleeper AI runs in the background looking for a winning/ending sequence of moves. If it spots a way to score a majority amount of boxes, 5 in our case (NN is only trained for 3 x 3 games), or a way to end the game it takes control. Otherwise, if no opportunity presents itself, it lies dormant.

How to play:
	python Game.py
		follow directions, player 1 is you (0), pick who to play (SB or just minimax alg)
	Making moves can be confusing open Move_Diagram.txt for refernce
		enter a two ints (ex. 6 2 is bottom right horizontal)

If you've never played Dots and Boxes: https://en.wikipedia.org/wiki/Dots_and_Boxes

Note: I coded everything from scratch, Neural Network included. The Neural Network isn't very good yet. I'm in the process of creating data and training. Looking to incorporate TensorFlow aswell, since their library runs much faster and has many useful features.