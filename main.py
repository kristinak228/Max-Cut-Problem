'''
Kristina R. Kolibab
Professor Tamon
Assignment 4 - Max Cut
Dec. 6th, 2018
'''

import picos as pic
import cvxopt as cvx
import networkx as nx
import numpy as np

# Read file and update graph
def add_edges_to_graph(G, infile):
	cnt = 1
	f = open(infile)
	n = sum(1 for line in open(infile))	
	for line in f:
		if(cnt == 1 or cnt == n):
			pass
		else:
			x, extra, y = line.split()
			y = y.strip(';')
			G.add_edge(x, y)
		cnt += 1

def main():
	# Basic starter stuff
	infile = input("Enter file name: ")
	N = int(input("Enter the number of vertices: "))
	outfile = input("Enter output file name: ")
	G = nx.Graph()
	add_edges_to_graph(G, infile)

	# Make G undirected
	G = nx.Graph(G)

	# Add weights to edges
	cnt_edges = 0
	for(i, j) in G.edges():
		G[i][j]['weight'] = 1
		cnt_edges += 1
#	print("Number of edges: %d" % cnt_edges)

	maxcut = pic.Problem()
	X = maxcut.add_variable('X', (N,N), 'symmetric')

	# Retrieve the Laplacian of the graph
	LL = 1/4.*nx.laplacian_matrix(G).todense()
	L = pic.new_param('L', LL)
	
	# Constrain X to have ones on the diagonal
	maxcut.add_constraint(pic.tools.diag_vect(X) == 1)
	# Constrain X to be positive semidefinite
	maxcut.add_constraint(X>>0)
	# Set object
	maxcut.set_objective('max', L|X)
	# Solve the problem
	maxcut.solve(verbose = 0, solver = 'cvxopt')		

	cvx.setseed(1)

	# Cholesky factorization
	V = X.value
	cvx.lapack.potrf(V)
	for i in range(N):
		for j in range(i+1,N):
			V[i,j] = 0

	# Do up to 100 projections
	cnt = 0
	obj_sdp = maxcut.obj_value()
#	print("Maxcut: %d" % (maxcut.obj_value())+1)
	obj = 0
	while(cnt < 100 or obj < 0.878*obj_sdp):
		r = cvx.normal(N, 1)
		x = cvx.matrix(np.sign(V*r))
		o = (x.T*L*x).value[0]
		if(o > obj):
			x_cut = x
			obj = o
		cnt += 1
	x = x_cut

	# Extract S and T separated nodes
	S1=[n for n in range(N) if x[n] < 0]
	S2=[n for n in range(N) if x[n] > 0]
	
	# Write the node cut to file
	fout = open(outfile, "w")
	fout.write("S Cut: ")
	for s1 in S1:
		fout.write(str(s1))
		fout.write(' ')
	fout.write('\n')
	fout.write("T Cut: ")
	for s2 in S2:
		fout.write(str(s2))
		fout.write(' ')
	fout.close()

	# Round up 1 for the probabilistic algorithm
	mc = maxcut.obj_value()+1
	print("Maxcut: %d, Number of Edges: %d" % (mc, cnt_edges))
	if(int(cnt_edges) == int(mc)):
		print("Maxcut equals the number of edges... This graph is bipartite!")

if __name__ == "__main__":
	main()




