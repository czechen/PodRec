#!/usr/bin/env python3

import dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


podcasts = dataset.Podcasts(200)
titles = podcasts.get_titles()

#get LSA representations of descriptions
Z = podcasts.LSA_representations(True)

#t-SNE embedding
Z_embedded = TSNE(n_components=2, learning_rate='auto',
	init='random',metric='correlation', perplexity=10).fit_transform(Z)
#plot the embedded system
print(type(Z_embedded))
print(Z_embedded.shape)
xs = []
ys = []
for k in Z_embedded:
	xs.append(k[0])
	ys.append(k[1])
import plotly.express as px
fig = px.scatter(x=xs,y=ys,text=titles)
fig.update_traces(mode="markers")
fig.show()

#PCA embedding
pca = PCA(n_components=2)
Z_embedded = pca.fit_transform(Z)
#plot the embedded system
print(type(Z_embedded))
print(Z_embedded.shape)
xs = []
ys = []
for k in Z_embedded:
	xs.append(k[0])
	ys.append(k[1])
import plotly.express as px
fig = px.scatter(x=xs,y=ys,text=titles)
fig.update_traces(mode="markers")
fig.show()
