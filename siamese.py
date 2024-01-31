#!/usr/bin/env python3


import dataset

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.python.framework.ops import disable_eager_execution
from random import shuffle

#disable_eager_execution()


#preprocesing

#define the training pairs (triples)


#similiarity metric used for training the siamese network
#two podcasts are similiar if they are the kth closest neighbor to each other
#or if they share the same categories
# NEEDS EXPERIMENTATION (hyperparameters: k in knn (viz. dataset.py))
def get_similiarity(podcast1,podcast2):
    knn_sim = 1/2*((podcast1.id in podcasts.knn_to_podcast(podcast2.id)) + (podcast2.id in podcasts.knn_to_podcast(podcast1.id)))
    if knn_sim >= 1/2:
        return knn_sim
    else:
        podcast1_categories = set(podcasts[podcast1.id].categories)
        podcast2_categories = set(podcasts[podcast2.id].categories)
        intersection = set.intersection(podcast1_categories,podcast2_categories)
        union = set.union(podcast1_categories,podcast2_categories)
        return len(intersection)/(len(union))

#load the word embeddings 
wv = KeyedVectors.load("genism_vw.vectors")

#embedding of the podcast description
def description_embedding(podcast):
    if not (hasattr(podcast,'embedded_description')):
        embedded_sequance = []
        for token in podcast.tokenized_description:
            embedded_sequance.append(wv[token])
        podcast.embedded_description = embedded_sequance
    return

#load the podcasts
podcasts = dataset.Podcasts(200)


#mapping function now unused
def map_data(desc1,desc2,dist):
    return {'input_1':desc1,'input_2':desc2},dist



BATCH_SIZE = 10

#generator for training dataset
def _input_fn(podcasts):
    def generator():
        for p1,p2 in zip(podcasts[:1000], np.random.permutation(podcasts[:1000])): #!!!!! only using part of the podcasts for testing
            if p1.id > p2.id:
                continue
            # Assuming embedded_description is a sequence of vectors
            description_embedding(p1)
            description_embedding(p2)
            desc1 = tf.ragged.constant(p1.embedded_description)
            desc2 = tf.ragged.constant(p2.embedded_description)
            # Pad the sequences to have the same length
            desc1 = desc1.to_tensor(default_value=0, shape=[111, 64])
            desc2 = desc2.to_tensor(default_value=0, shape=[111, 64])
            yield (desc1,desc2),get_similiarity(p1,p2)

    dataset = tf.data.Dataset.from_generator(generator,output_signature=(
        (tf.TensorSpec(shape=(111,64), dtype=tf.float32),
        tf.TensorSpec(shape=(111,64), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.float32)))
    #dataset = dataset.map(map_data)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

dataset = _input_fn(podcasts)

#model (viz. https://ojs.aaai.org/index.php/AAAI/article/view/10350)

#!!!!!! NEEDS TUINING !!!!!!!!!

rnn_units = 64
LSTM = tf.keras.layers.LSTM(rnn_units) 

# Define model inputs
input_1 = tf.keras.Input(shape=(111,64), ragged=False, name='input_1')
input_2 = tf.keras.Input(shape=(111,64), ragged=False, name='input_2')

# Define model operations
output_1 = LSTM(input_1)
output_2 = LSTM(input_2)
distance = tf.math.exp(-tf.math.abs(output_1 - output_2))


#training
tf.compat.v1.experimental.output_all_intermediates(True)
# Create and compile model
model = tf.keras.Model(inputs=[input_1, input_2], outputs=distance)
model.summary()
tf.keras.utils.plot_model(model,'model.png',show_shapes=True)
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
model.fit(dataset,epochs=10)



#TODO - implement embedding of podcast description and searching with knn to evaluate the model
#TODO (maybe) - embedding to 2d space (viz. LSA)