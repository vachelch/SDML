import numpy as np 
import random

# validation set
train_path = "t2-train.txt"
test_path = "t2-test.txt"

dir = './'

set_train = set()

def read_data(file_path, st):
	data = []
	with open(file_path, 'r') as f:
		for row in f.readlines():
			s, t = row.split()
			s = int(s)
			t = int(t)

			st.add(s)
			st.add(t)
			data.append([s, t])
	return data

def get_graph(edges):
	adj_list = {}
	for t, s in edges:
		if s not in adj_list:
			adj_list[s] = set([t])	
		else:
			adj_list[s].add(t)

	return adj_list

data = read_data(train_path, set_train)

# 10 persent as val node, get train graph, val graph
nodes = list(set_train)
val_nodes = nodes[: int(0.01*len(nodes))]
val_nodes = set(val_nodes)
train_nodes = nodes[int(0.01*len(nodes)): ]
train_nodes = set(train_nodes)

train_edges = []
val_edges = []

for s, t in data:
	if (s in val_nodes) or (t in val_nodes):
		val_edges.append([s, t])
	else:
		train_edges.append([s, t])

# construct graph
adj_list = {}
adj_list = get_graph(train_edges)

mean_degree = len(train_edges) / len(train_nodes)

# degree
degrees = [mean_degree for i in range(17500)]
for key, adjs in adj_list.items():
	degrees[key - 1] = len(adjs)

# common intersection and the union
intersection = 0
union = 0
for s, t in train_edges:
	if s in adj_list and t in adj_list:
		intersection += len(adj_list[s].intersection(adj_list[t]))
		union += len(adj_list[s].union(adj_list[t]))
	else:
		if s in adj_list:
			union += len(adj_list[s])
		else:
			union += len(adj_list[t])

mean_intersect = intersection / len(train_edges)
mean_union = union/ len(train_edges)

np.save(dir + 'graph_feature.npy', [degrees, adj_list, mean_intersect, mean_union])
print("mean_degree, mean_intersect, mean_union", mean_degree, mean_intersect, mean_union)


#

























































