import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from copy import deepcopy

from src.data_preparation import *
from src.modelling_nn_new import *
from src.utils import *

###############################################################################
# Distrbuted setting
NN = 5

# Dataset Settings
TARGET = 7
TRAIN_AGENT_DATA = 256
TEST_AGENT_DATA = 100
TRAIN_SIZE = TRAIN_AGENT_DATA * NN
TEST_SIZE = TEST_AGENT_DATA * NN

# Training parameters
N_EPOCHS = 500
STEP_SIZE = 5e-4
BATCH_SIZE = 64
N_BATCH = int(TRAIN_AGENT_DATA / BATCH_SIZE)

# saving 


###############################################################################
# os.mkdir(img_dir)
fix_seed(42)

## DATASET PREPARATION
x_train, y_train, x_test, y_test = prepare_dataset("./dataset/",
                                                    TARGET,
                                                    TRAIN_SIZE,
                                                    TEST_SIZE,
                                                    (int(np.sqrt(input_shape)), int(np.sqrt(input_shape))))

x_train, y_train = split_images_per_agents(x_train, y_train, NN, TRAIN_AGENT_DATA)
x_test, y_test = split_images_per_agents(x_test, y_test, NN, TEST_AGENT_DATA)
print(f"\nTrain data size: {x_train.shape}")
print(f"\nTrain laebl size: {y_train.shape}")
print(f"\nTest data size: {x_test.shape}")
print(f"\nTest laebl size: {y_test.shape}")

for i in range(NN):
    print(Counter(y_train[i]))
# exit()

print(f"\nTrain size: {x_train.shape}")

#  Generate Network Graph
G = generate_graph(NN, graph_type="Cycle")
# G = generate_graph(N_AGENTS, graph_type="Path")
# G = generate_graph(N_AGENTS, graph_type="Star")

ID_AGENTS = np.identity(NN, dtype=int)

while 1:
	ADJ = nx.adjacency_matrix(G)
	ADJ = ADJ.toarray()	

	test = np.linalg.matrix_power((ID_AGENTS+ADJ),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()

# METROPOLIS HASTING
WW = metropolis_hasting(ADJ, NN)

print('Row Stochasticity {}'.format(np.sum(WW,axis=1)))
print('Col Stochasticity {}'.format(np.sum(WW,axis=0)))

# Network Variables
uu_init = []
for l in range(T-1):
    uu_init.append(np.random.randn(layer_neurons[l+1], layer_neurons[l]+1)/10)

uu = [deepcopy(uu_init) for _ in range(NN)]
uu_kp1 = deepcopy(uu)

ss_init = []
for l in range(T-1):
    ss_init.append(np.zeros((layer_neurons[l+1], layer_neurons[l]+1)))
ss = [deepcopy(ss_init) for _ in range(NN)]
ss_kp1 = deepcopy(ss)

grads_init = []
for l in range(T-1):
    grads_init.append(np.zeros((layer_neurons[l+1], layer_neurons[l]+1)))
                          
grads = [deepcopy(grads_init) for _ in range(NN)]

# plot variables
u_err = np.zeros((N_EPOCHS, NN))
J = np.zeros((N_EPOCHS, NN)) # Cost function
NormGradJ = np.zeros((N_EPOCHS, NN))
consensus_erros = np.zeros((N_EPOCHS, NN))

# Init of Gradient Tracking Algorithm
for agent in range(NN):
    for img in range(BATCH_SIZE):                        
        xx = forward_pass(uu[agent], x_train[agent, img])
        loss, loss_grad = binary_cross_entropy(xx[-1][0], y_train[agent, img])
        
        _, grad_kp1 = backward_pass(xx, uu[agent], loss_grad)
        
        for l in range(T-1):
            grads[agent][l] += grad_kp1[l]

    for l in range(T-1):
        ss[agent][l] = grads[agent][l]

print()
###############################################################################
# TRAIN
for epoch in range(N_EPOCHS):
    for batch_iter in range(N_BATCH):
        for agent in range(NN):

            neighs = np.nonzero(ADJ[agent])[0]

            # Gradient Tracking Algorithm - Weights Update
            for l in range(T-1):
                uu_kp1[agent][l] = (WW[agent, agent] * uu[agent][l]) - (STEP_SIZE * ss[agent][l])
                for neigh in neighs:
                    uu_kp1[agent][l] += WW[agent, neigh] * uu[neigh][l]

            batch_grads_kp1 = [np.zeros_like(u_l) for u_l in uu[agent]]
            for i in range(BATCH_SIZE):
                img = batch_iter*BATCH_SIZE + i 
            
                # Forward pass
                xx = forward_pass(uu_kp1[agent], x_train[agent, img])

                # Loss evalutation
                loss, loss_grad = binary_cross_entropy(xx[-1][0], y_train[agent, img])
                J[epoch, agent] += loss / TRAIN_AGENT_DATA
                
                # Backward pass
                _, grad_kp1 = backward_pass(xx, uu_kp1[agent], loss_grad)
                for l in range(T-1):
                    batch_grads_kp1[l] += grad_kp1[l] / BATCH_SIZE

                for l in range(T-1):
                    NormGradJ[epoch, agent] += (np.abs(grad_kp1[l]).sum() / grad_kp1[l].size) / TRAIN_AGENT_DATA

            # Gradient Tracking Algorithm - SS Update
            for l in range(T-1):
                ss_kp1[agent][l] = (WW[agent, agent] * ss[agent][l]) + (batch_grads_kp1[l] - grads[agent][l])
                for neigh in neighs:
                    ss_kp1[agent][l] += WW[agent, neigh] * ss[neigh][l]
            
            for l in range(T-1):
                grads[agent][l] = batch_grads_kp1[l] / N_BATCH

    # SYNCHRONOUS UPDATE
    for agent in range(NN):
        for l in range(T-1):
            uu[agent][l] = uu_kp1[agent][l]
            ss[agent][l] = ss_kp1[agent][l] 

    uu_mean = [np.zeros_like(u_l) for u_l in uu[0]]
    for l in range(T-1):
        for agent in range(NN):
            uu_mean[l] += uu[agent][l] / 5

    for agent in range(NN):
        for l in range(T-1):
            consensus_erros[epoch, agent] += np.abs(uu_mean[l] - uu[agent][l]).sum() / uu_mean[l].size
    
    if epoch % 1 == 0:
        print(f'Iteration n° {epoch:d}: loss = {np.mean(J[epoch]):.4f}, grad_loss = {np.mean(NormGradJ[epoch]):.4f}')

print(f'Iteration n° {epoch:d}: loss = {np.mean(J[epoch]):.4f}, grad_loss = {np.mean(NormGradJ[epoch]):.4f}') # last iteration

# Computes the mean error over uu
print("\nAgent erros:")
uu_mean = [np.zeros_like(u_l) for u_l in uu[0]]
for l in range(T-1):
    for agent in range(NN):
        uu_mean[l] += uu[agent][l] / 5

u_err = np.zeros(NN)
for agent in range(NN):
    for l in range(T-1):
        u_err[agent] += np.linalg.norm(uu_mean[l] - uu[agent][l])
    print(f' - Agent {agent} mean_error = {u_err[agent]}')


###############################################################################
# TEST
if 1:
    agent = 0
    good = 0
    for img in range(TRAIN_AGENT_DATA):
        # in this way, as we did for computation of BCE, we got only the output value of the network
        pred = round(forward_pass(uu[agent], x_train[agent, img])[-1] [0])
        if pred == y_train[agent, img]:
            good +=1
                
    print(f"\nAccuracy of agent 0 on train set: {good/TRAIN_AGENT_DATA*100:.2f} %")

if 1:
    agent = 0
    good = 0
    for img in range(TEST_AGENT_DATA):
        # in this way, as we did for computation of BCE, we got only the output value of the network
        pred = round(forward_pass(uu[agent], x_test[agent, img])[-1][0])
        if pred == y_test[agent, img]:
            good +=1
                        
    print(f"Accuracy of agent 0 on test set: {good/TEST_AGENT_DATA*100:.2f} %")


###############################################################################
# PLOT

img_dir = os.path.join("./imgs/task13/", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(img_dir)

plt.figure('Cost function', figsize=(12,8))
plt.title('Evolution of the cost function')
plt.semilogy(range(N_EPOCHS),np.sum(J, axis=1)/NN, label='Total Normalized Cost Evolution', linewidth = 2)
for agent in range(NN):
     plt.semilogy(range(N_EPOCHS), J[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r'$\sum_{n=1}^N J$')
plt.legend()
plt.savefig(f"{img_dir}/cost_funct.png", )

plt.figure('Norm of Gradient function', figsize=(12,8))
plt.title('Evolution of the norm of the gradient of the cost function')
plt.semilogy(range(N_EPOCHS), np.sum(NormGradJ, axis=-1)/NN, label='Total Norm Gradient Evolution', linewidth = 2)
for agent in range(NN):
    plt.semilogy(range(N_EPOCHS), NormGradJ[:, agent], linestyle = ':')
plt.xlabel(r'Epochs')
plt.ylabel(r"$|| \nabla J_w(x_t^k) ||_2$")
plt.legend()
plt.savefig(f"{img_dir}/norm_grad_cost.png")

plt.figure('Evolution of the  error', figsize=(12,8))
plt.title('Evolution of agent weights error from the mean value')
for agent in range(NN):
    plt.semilogy(range(N_EPOCHS), consensus_erros[:, agent], label=f"agent{agent+1}")
plt.xlabel('Updates')
plt.ylabel(r'$|| u^\star - u_i ||_2, \quad u^\star = \frac{1}{N} \sum_{i=1}^N u_i \quad \forall i=1,\dots,N $')
plt.legend(loc="upper right")
plt.savefig(f"{img_dir}/consensus_erros.png")

# plt.show()