#!/usr/bin/env python3

from gensim.models import Word2Vec
from gensim.models import Phrases
from keras.preprocessing.text import text_to_word_sequence

import dataset


def train_word2vec(documents,phrases=False,embedding_dim=128):
	if phrases:
		documents = Phrases(documents)
	print('Training..')
	model = Word2Vec(documents,vector_size=embedding_dim, window=5, min_count=1, workers=4)
	word_vectors = model.wv
	word_vectors.save('genism_vw.vectors')
	del model
	return word_vectors


if __name__ == "__main__":
	#define a class Corpus for word embeddings containg the podcast descriptions
	class Corpus(object):
		def __init__(self):
			podcasts = dataset.Podcasts(200)
			self.desc = podcasts.get_descriptions()

		def __iter__(self):
			for des in self.desc:
				yield text_to_word_sequence(
			    des,
			    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
			    lower=True,
			    split=' ')
	

	sentances = Corpus()
	#embed = train_word2vec(sentances,False,64) #uncomment to generate the word embeddings
