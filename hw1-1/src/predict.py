from scipy import spatial
import numpy as np
import pickle

file_path = 'embeddings'
ini2idx = pickle.load(open('ini2idx', 'rb'))

def get_embedding(file):
        embedding = {}
        with open(file_path, 'r') as f:
                f.readline()
                for row in f.readlines():
                        row = row.split()
                        row = [float(num) for num in row]
                        embedding[row[0]] = row[1:]
        return embedding

embedding = get_embedding(file_path)

def read_test(file):
        s = []
        t = []
        with open(file, 'r') as f:
                for row in f.readlines():
                        s_num, t_num = row.split()
                        s_num = int(s_num)
                        t_num = int(t_num)

                        s.append(s_num)
                        t.append(t_num)

        return s, t


# predict
s, t = read_test('t1-test.txt')
preds = []
for i in range(len(s)):
        if s[i] not in ini2idx or t[i] not in ini2idx:
                preds.append(1)
        else:
                s_num = ini2idx[s[i]]
                t_num = ini2idx[t[i]]

                if s_num not in embedding or t_num not in embedding:
                        preds.append(1)
                else:
                        cos = 1 - spatial.distance.cosine(embedding[s_num], embedding[t_num])
                        if cos > 0.63: preds.append(1)
                        else: preds.append(0)

with open('pred.txt', 'w') as f:
        for p in preds:
                f.write("%d\n"%(p))

print(np.sum(preds) / len(preds))

