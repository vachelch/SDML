import numpy as np 
import random, math

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC

np.random.seed = 1024

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
	for s, t, _ in edges:
		if s not in adj_list:
			adj_list[s] = set([t])	
		else:
			adj_list[s].add(t)

	return adj_list

def get_negtive(train_adj, val_adj, length):
	num = 0
	negtive_data = []
	while(1):
		for s, adjs in train_adj.items():
			# random t
			adjs = list(adjs)
			idx = random.randint(0, len(adjs)-1)
			t = adjs[idx]
			if t in train_adj:
			# random n
				adjs = list(train_adj[t])
				idx = random.randint(0, len(adjs)-1)
				n = adjs[idx]

				if (n not in train_adj[s]) and ((s not in val_adj) or (n not in val_adj[s])):
					negtive_data.append([s, n, 0])
					num += 1

					if (num == length): return negtive_data

# def idx2vec(train_x_idx, texts_tfidf, times = None):
# 	train_x = []
# 	for s, t in train_x_idx:
# 		s_vec = list(texts_tfidf[s-1])
# 		t_vec = list(texts_tfidf[t-1])
# 		# vec = np.concatenate((s_vec, t_vec))
# 		# vec = np.array(s_vec) - np.array(t_vec)
# 		vec = (np.array(s_vec) * np.array(t_vec)).sum()
# 		if times is not None:
# 			train_x.append([vec, times[s-1] - times[t-1], times[s-1] + times[t-1]])
# 		else:
# 			train_x.append([vec])

# 	return train_x

def idx2vec(train_x_idx, texts_tfidf, graph, times = None):
	train_x = []

	for s, t in train_x_idx:
		s_vec = texts_tfidf[s-1]
		t_vec = texts_tfidf[t-1]
		# # high dimension td-idf score
		tfidf_socre = (s_vec * t_vec).sum() / (math.sqrt((s_vec**2).sum()) * math.sqrt((t_vec**2).sum()))

		
		# degree
		degrees, adj_list_cited, mean_intersect, mean_union = graph
		s_degree = degrees[s-1]
		t_degree = degrees[t-1]

		# # common neighbors
		# if s in adj_list_cited and t in adj_list_cited:
		# 	intersect = len(adj_list_cited[s].intersection(adj_list_cited[t]))
		# else:
		# 	intersect = mean_intersect

		# # union
		# if s in adj_list_cited and t in adj_list_cited:
		# 	union = len(adj_list_cited[s].union(adj_list_cited[t]))
		# else:
		# 	union = mean_union

		# # jaccard's coeff
		# jaccard = intersect / union

		# word embedding
		# s_emb = doc_embs[s-1]
		# t_emb = doc_embs[t-1]
		# emb_score = (s_emb * t_emb).sum() / (math.sqrt((s_emb**2).sum()) * math.sqrt((t_emb**2).sum()))
		# emb = s_emb - t_emb

		# time one hot
		if times is not None:
			time_diff = int(times[s - 1] - times[t-1])
			time_diff_one_hot = np.zeros(13)
			time_diff_one_hot[time_diff] = 1

		# get feature vector
		if times is not None:
			# train_x.append(np.concatenate(([tfidf_socre, times[s-1], times[t-1] , times[s-1] - times[t-1], times[s-1] + times[t-1], s_degree, t_degree, s_degree + t_degree], s_emb, t_emb, emb)))
			# , s_degree, t_degree, s_degree + t_degree, s_degree - t_degree, intersect, union, jaccard
			train_x.append([tfidf_socre, times[s-1], times[t-1] , times[s-1] - times[t-1], times[s-1] + times[t-1], s_degree, t_degree, s_degree + t_degree])
		else:
			train_x.append([tfidf_socre, s_degree, t_degree, s_degree + t_degree])

	return train_x

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
		val_edges.append([s, t, 1])
	else:
		train_edges.append([s, t, 1])


texts_tfidf = np.load(dir+"texts_tfidf.npy")
graph = np.load(dir+"graph_feature.npy")

# construct graph
train_adj = {}
val_adj = {}
train_adj = get_graph(train_edges)
val_adj = get_graph(val_edges)


# split data
negtive_train = get_negtive(train_adj, val_adj, len(train_edges))
negtive_val = get_negtive(val_adj, train_adj, len(val_edges))

train = np.concatenate((np.array(train_edges), np.array(negtive_train)), axis= 0)
val = np.concatenate((np.array(val_edges), np.array(negtive_val)), axis= 0)

np.random.shuffle(train)
np.random.shuffle(val)


train_x_idx = train[:, :2]
train_y = train[:, -1]

val_x_idx = val[:, :2]
val_y = val[:, -1]

# integrate features
train_x = idx2vec(train_x_idx, texts_tfidf, graph)
val_x = idx2vec(val_x_idx, texts_tfidf, graph)


train_x = np.array(train_x)
val_x = np.array(val_x)

# train
# clf = RandomForestClassifier(n_estimators=500, min_samples_leaf = 5, random_state=0, n_jobs= -1)
clf = RandomForestRegressor(n_estimators=500, min_samples_leaf = 40, random_state=0, n_jobs= -1)
# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter = 500)
clf.fit(train_x, train_y)
preds_train = clf.predict(train_x)
preds = clf.predict(val_x)

train_acc = np.equal(np.round(preds_train), train_y).sum() / len(train_y)
test_acc = np.equal(np.round(preds), val_y).sum() / len(val_y)

print('train_acc: ', train_acc)
print('val_acc: ', test_acc)
print('val Pos/neg ratio', sum(preds) / len(preds))


# test
set_test = set()
test_idx = read_data(test_path, set_test)
test_x = idx2vec(test_idx, texts_tfidf, graph)
preds = clf.predict(test_x)

count = 0
with open('pred.txt', 'w') as f:
	for pred in preds:
		if pred > 0.7:
			count += 1
			f.write("%d\n"%(1))
		else:
			f.write("%d\n"%(0))

print('test Pos/neg ratio', count / len(preds))










