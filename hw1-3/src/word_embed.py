import gensim
import  re
import numpy as np 
import math
import _pickle as cPickle

# by training
dir_name = 't3-doc'
dir = './'
emb_file = 'wiki.en.vec'

def get_texts(dir_name):
	texts = []
	for i in range(1, 17500 + 1):
		file = dir_name + '/' + str(i) + '.xml'
		with open(file, 'r') as f:
			html = f.read()
			html = html.replace("\n", "")

		title = re.search(r'<title>(.*)</title>', html).group(1).lower()
		abstract = re.search(r'<abstract>(.*)</abstract>', html).group(1).lower()

		texts.append(title.strip() + '\n' + abstract.strip())

	return texts


def to_vector(texts):
	texts_vec = []
	for row in texts:
		texts_vec.append(row.split())

	return texts_vec


def get_doc_emb(texts_vec, emb_path, texts_tfidf, word_index):
	emb_model = gensim.models.KeyedVectors.load_word2vec_format(emb_path)

	mean_embs = []


	for i, row in enumerate(texts_vec):
		mean_emb = np.zeros(300)
		row_tfidf = texts_tfidf[i]

		words = []
		for word in set(row):
			if word in emb_model and word in word_index:
				words.append([ word, row_tfidf[word_index[word]] ])

		words = sorted(words, key = lambda w: w[1], reverse = True)
		
		for word in words[:10]:
			mean_emb += np.array(emb_model[word])

		mean_embs.append(mean_emb / 10)

	return mean_embs

# def get_doc_emb(texts_vec, emb_path):
# 	emb_model = gensim.models.KeyedVectors.load_word2vec_format(emb_path)

# 	mean_embs = []

# 	for i, row in enumerate(texts_vec):
# 		mean_emb = np.zeros(300)
# 		count = 0

# 		for word in row:
# 			if word in emb_model:
# 				mean_emb += emb_model[word]
# 				count += 1

# 		mean_emb /= count
# 		mean_embs.append(mean_emb)

# 	return mean_embs

texts = get_texts(dir_name)
texts_vec = to_vector(texts)
texts_tfidf = np.load(dir+ 'texts_tfidf.npy')
texts_tfidf = np.array(texts_tfidf, dtype = np.float32)
word_index = cPickle.load(open(dir + 'word_index.pkl', 'rb'))

doc_embs = get_doc_emb(texts, dir + emb_file, texts_tfidf, word_index)
np.save(dir + 'doc_embs.npy', doc_embs)





















