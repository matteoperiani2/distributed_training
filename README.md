# Distributed Training of Neural Networks
Implementation of the gradient tracking algorithm to training a network of N agents that implement each a neural network over a simply classification task with the Fashion MNIST dataset. The implemntation of the neural network is very simple and inefficent, this code is used as project for university exam. The focus was not creating a good neural network to achive good result, but instead to understand how the training procedure can be distributed over multple cooperating agents.

In the file located in "src/modeling_nn.py" you can set the size of the network that currently is [FLAT_IMG_SIZE, 32, 1] (binary task) and on the file "main.py" yoy can choose:
 1) NN: the number of agents
 2) N_EPOCHS: number of epocs
 3) BATCH_SIZE: size of the batch:
 4) STEP_SIZE: the learning rate
 5) the size of the train and the test set
