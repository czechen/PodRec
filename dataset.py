#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from keras.preprocessing.text import text_to_word_sequence
from itertools import count

def tokenizer(text):
    return text_to_word_sequence(
                text,
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                lower=True,
                split=' ')


@dataclass
class Podcast(object):
	"""
	Class for informations (title, description, rss url) about each podcast
	"""
	title: str
	description: str
	categories: list
	rss_url: str
	_ids = count(0)
	def __post_init__(self):
		self.tokenized_description = tokenizer(self.description)
		self.id = next(self._ids)

	def __str__(self):
		return f'Title: {self.title}; Descritpion: {self.description}; Categories: {self.categories}; rss_url: {self.rss_url}'

class Podcasts(object):
	"""
	Podcasts data set class. During initialization it loads the csv file, perform some basic filtration and stores Podcasts in self.data

	Methods:
		get_descriptions - returns a list of description of podcats in the same order as self.data
		get_titles - returns a list of titles of podcats in the same order as self.data
		search_by_title - allows for search by substring in Podcast.title, returns a list of matched podcats
	"""

	def __init__(self,min_descrpition_len):
		self.data = []
		titles = []
		with open('final_podcasts_en_processed.csv',newline='') as cvs:
			rows = csv.reader(cvs)
			next(rows) #skip header
			for row in rows:
				category = row[-2].split(',')
				description = row[4].lower()
				title = row[0].lower()
				rss = row[5]
				if (len(description) < min_descrpition_len) or (title in titles):
					continue #ignore podcasts with short description
				else:
					titles.append(title)
					self.data.append(Podcast(title,description,category,rss))

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_descriptions(self):
			return [i.description for i in self]
	def get_titles(self):
			return [i.title for i in self]
	def search_by_title(self,title):
		match = []
		for pod in self:
			if title in pod.title:
				match.append(pod)
		return match

	def lemmatize_descriptions(self):
		from nltk.stem import WordNetLemmatizer
		WNL = WordNetLemmatizer()
		for podcast in self:
			podcast.lemmatized_description = [WNL.lemmatize(i) for i in podcast.tokenized_description]

	def LSA_representations(self,lemmatize=True):
		from sklearn.feature_extraction import text
		from sklearn.decomposition import TruncatedSVD

		self.vectorizer = text.TfidfVectorizer()
		if lemmatize:
			if not (hasattr(self[0], 'lemmatized_description')):
				self.lemmatize_descriptions()
			corpus = [" ".join(podcast.lemmatized_description) for podcast in self]
		else:
			corpus = self.get_descriptions()
		corpus_tfidf = self.vectorizer.fit_transform(corpus)
		svd = TruncatedSVD(n_components=100,algorithm='arpack')
		self.corpus_svd = svd.fit_transform(corpus_tfidf)
		for index in range(len(self.data)):
			self.data[index].LSA_representation = self.corpus_svd[index]
		return self.corpus_svd

	def LSA_KNN(self,k=10):
		'''
		functions which computes kNN based on LSA representations
		'''
		from sklearn.neighbors import NearestNeighbors

		if not (hasattr(self,'corpus_svd')):
			self.LSA_representations()
		self.knn = NearestNeighbors(n_neighbors=k,leaf_size=100,metric='correlation')
		self.knn.fit(self.corpus_svd)

	def knn_to_podcast(self,index,k=10):
		if not (hasattr(self,'knn')):
			self.LSA_KNN(k)
		return self.knn.kneighbors([self.corpus_svd[index]],return_distance=False)[0]

if __name__ == "__main__":
	import numpy as np
	Podcasts = Podcasts(200)
	des_len = [len(d)for d in Podcasts.get_descriptions()]
	print(len(Podcasts))
	print(np.mean(des_len),min(des_len))
	while(True):
		'''
		search = input("title:")
		for pod in Podcasts.search_by_title(search):
			print(pod)
		'''
		search_index = int(input('index:'))
		print(f'Searching knn to {Podcasts[search_index].title}')
		for k_th,neighbor_index in enumerate(np.unique(Podcasts.knn_to_podcast(search_index,10))):
			if neighbor_index == search_index:
				continue
			print(neighbor_index)
			print(f'{k_th}: {Podcasts[neighbor_index].title}')