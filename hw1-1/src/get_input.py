import numpy as np 
import random
import pickle

ini2idx = {}
data = []
# read training file
with open('t1-train.txt', 'r') as f:
	for row in f.readlines():
		s, t = row.split()
		s = int(s)
		t = int(t)
		data.append([s, t])

		if s not in ini2idx:
			ini2idx[s] = len(ini2idx)

		if t not in ini2idx:
			ini2idx[t] = len(ini2idx)

# read testing file
with open('t1-test-seen.txt', 'r') as f:
	for row in f.readlines():
		s, t = row.split()
		s = int(s)
		t = int(t)
		data.append([s, t])

		if s not in ini2idx:
			ini2idx[s] = len(ini2idx)

		if t not in ini2idx:
			ini2idx[t] = len(ini2idx)

data_idx = [[ini2idx[s], ini2idx[t]] for s, t in data]

with open('data.txt', 'w') as f:
	for s, t in data_idx:
		f.write("%d %d\n"%(s, t))


train_edge = data_idx[:int(len(data_idx) * (0.8 + 0.2 * 0.8))]
test_edge = data_idx[int(len(data_idx) * (0.8 + 0.2 * 0.8)) : ]
with open('train_data.txt', 'w') as f:
	for s, t in train_edge:
		f.write("%d %d\n"%(s, t))

with open('test_data.txt', 'w') as f:
	for s, t in test_edge:
		f.write("%d %d\n"%(s, t))


test_x = []
test_y = []
for s, t in test_edge:
	test_x.append([s, t])
	test_y.append(1)
	idx = random.randint(0, len(train_edge)-1)

	neg_t = train_edge[idx][0]
	test_x.append([s, neg_t])
	test_y.append(0)

np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)
pickle.dump(ini2idx, open('ini2idx', 'wb'))
		
# deepwalk --format edgelist --input data.txt --max-memory-data-size 0 --number-walks 80 --representation-size 128 --walk-length 40 --window-size 10 --workers 20 --output embeddings


























