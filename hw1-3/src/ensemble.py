import numpy as np 

method = 'rf_'
preds = np.load(method + '0.npy')

for i in range(1, 7):
	tmp = np.load(method + str(i) + '.npy')
	preds += tmp

with open('pred.txt', 'w') as f:
	for p in preds:
		if p >= 4:
			f.write("%d\n"%(1))
		else:
			f.write("%d\n"%(0))