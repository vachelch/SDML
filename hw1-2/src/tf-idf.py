import _pickle as cPickle
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
import  re
import numpy as np 
import math

dir_name = 't2-doc'
dir = './'

def get_texts(dir_name):
	texts = []
	for i in range(1, 17500 + 1):
		file = dir_name + '/' + str(i) + '.xml'
		with open(file, 'r') as f:
			html = f.read()
			html = html.replace("\n", "")

		title = re.search(r'<title>(.*)</title>', html).group(1)
		abstract = re.search(r'<abstract>(.*)</abstract>', html).group(1)

		text = title + " " + abstract
		texts.append(text)

	return texts

def get_test(file_path):
	test_x = []
	with open(file_path, 'r') as f:
		for row in f.readlines():
			s, t = row.split()
			s = int(s)
			t = int(t)

			test_x.append([s, t])

	return test_x

def similar_score(s, t, texts_tfidf):
	s_vec = texts_tfidf[s-1]
	t_vec = texts_tfidf[t-1]

	score = sum(s_vec * t_vec) / (math.sqrt(np.sum(s_vec**2)) * math.sqrt(np.sum(s_vec**2)) )
	# score = sum(s_vec * t_vec)

	return score


texts = get_texts(dir_name)
tokenizer = Tokenizer(num_words = 10000, 
       filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890', 
       lower=True, split=' ', 
       char_level=False, 
       oov_token=None)

tokenizer.fit_on_texts(texts)
texts_tfidf = tokenizer.texts_to_matrix(texts, mode='tfidf')
np.save(dir + 'texts_tfidf.npy', texts_tfidf)

cPickle.dump(tokenizer.word_index, open(dir + 'word_index.pkl', 'wb'))

# tf-idf penalized by document length, okapi(too much)
# texts_bow = tokenizer.texts_to_matrix(texts, mode='freq')
# texts_binary = tokenizer.texts_to_matrix(texts, mode='binary')

# N = len(texts_bow)
# tf_raw = texts_bow
# df_raw = texts_binary.sum(0)
# dl = texts_bow.sum(1)
# avdl = dl.sum() / N

# tfs = ((1 + 1.5) * tf_raw / (1.5*(1 - 0.75 + 0.75 * dl[:, np.newaxis] / avdl) + tf_raw))
# dfs = np.log((N - df_raw + 0.5) / (df_raw + 0.5))
# texts_tfidf = tfs * dfs



# # read test
# test_path = 't3-test.txt'
# test_x = get_test(test_path)

# # predict
# similarities = []

# for s, t in test_x:
# 	score = similar_score(s, t, texts_tfidf)
# 	similarities.append(score)

# tmp_arr = sorted(similarities)
# threshold = tmp_arr[len(tmp_arr) // 2]

# print(threshold)
# # print(tmp_arr[-1000:-1])


# ones = 0
# # output
# with open('pred_0.8.txt', 'w') as f:
# 	for score in similarities:
# 		if score > threshold:
# 			f.write("%d\n"% (1))
# 			ones += 1
# 		else:
# 			f.write("%d\n"% (0))

# print(ones)
# print(ones / len(test_x))
























