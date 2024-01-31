#!/usr/bin/env python3


import dataset

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.python.framework.ops import disable_eager_execution
from random import shuffle
from scipy.spatial.distance import correlation as corr_metric
from sklearn.manifold import TSNE
from keras import backend as K
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors



image_model = tf.keras.models.load_model('model.h5')
lstm_layer = image_model.layers[2]
#rnn_units = 32
#new_lstm = tf.keras.layers.LSTM(rnn_units) 
#new_lstm.set_weights(lstm_layer.get_weights())
#lstm.return_sequences=True

#load the word embeddings 
wv = KeyedVectors.load("genism_vw.vectors")
def description_embedding(podcast):
    if not (hasattr(podcast,'embedded_description')):
        embedded_sequance = []
        for token in podcast.tokenized_description:
            embedded_sequance.append(wv[token])
        podcast.embedded_description = embedded_sequance
    return

input_data = tf.keras.Input(shape=(111,64), ragged=False)
out = lstm_layer(input_data)
model = tf.keras.Model(inputs=input_data, outputs=out)

model.compile(optimizer='adam', loss='mse')
#func = tf.keras.backend.function([input_data], [new_lstm.output])


vector_representation = []
podcasts = dataset.Podcasts(200)
titles = podcasts.get_titles()
for pod in podcasts:
	description_embedding(pod)
	desc1 = tf.ragged.constant(pod.embedded_description)
	desc1 = desc1.to_tensor(default_value=0, shape=[111, 64])
	vector_representation.append(desc1)
vector_representation = np.array(vector_representation)
vector_representation = model.predict(vector_representation )
#vector_representation = func(vector_representation)


knn = NearestNeighbors(n_neighbors=10,leaf_size=100,metric='correlation')
knn.fit(vector_representation)
huber = knn.kneighbors([vector_representation[0]],return_distance=False)[0]
smartless = knn.kneighbors([vector_representation[1]],return_distance=False)[0]
print('huber:')
for k_th,neighbor_index in enumerate(huber):
	
	if neighbor_index == 0:
			continue
	print(neighbor_index)
	print(f'{k_th}: {podcasts[neighbor_index].title}')
print('smartless:')
for k_th,neighbor_index in enumerate(smartless):
	if neighbor_index == 1:
			continue
	print(neighbor_index)
	print(f'{k_th}: {podcasts[neighbor_index].title}')

vector_representation_embedded = TSNE(n_components=2,n_iter=2000,n_iter_without_progress=500, learning_rate='auto',
	init='random', random_state=42,metric='correlation', perplexity=10).fit_transform(vector_representation)
#plot the embedded system
print(type(vector_representation_embedded))
print(vector_representation_embedded.shape)
xs = []
ys = []
for k in vector_representation_embedded:
	xs.append(k[0])
	ys.append(k[1])
import plotly.express as px
figure = px.scatter(x=xs,y=ys,text=titles)
figure.update_traces(mode="markers")
figure.show()

fig,ax  = plt.subplots(figsize = (8,10),dpi=300)
ax.scatter(xs,ys,s=0.5,color='k')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('siamese.jpg')

