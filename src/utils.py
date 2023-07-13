import random
import numpy as np
import networkx as nx

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_graph(n_agents, graph_type = "Cycle"):
	if graph_type == "Cycle":
		G = nx.path_graph(n_agents)
		G.add_edge(n_agents-1,0)

	if graph_type == "Path":
		G = nx.path_graph(n_agents)
		
	if graph_type == "Star":
		G = nx.star_graph(n_agents-1)

	return G
	
def get_adj_matrix(G, NN):
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
	return ADJ

def metropolis_hasting(adj, n_agents):
	degree = np.sum(adj, axis=0)
	WW = np.zeros((n_agents,n_agents))

	for ii in range(n_agents):
		Nii = np.nonzero(adj[ii])[0]
		
		for jj in Nii:
			WW[ii,jj] = 1/(1+np.max([degree[ii],degree[jj]]))

		WW[ii,ii] = 1-np.sum(WW[ii,:])

	return WW

def quadratic_fn(x,Q,r):
	fval = 0.5*x@Q@x+r@x
	fgrad = Q@x+r

	return fval, fgrad
