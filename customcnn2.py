
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
# -*- coding: utf-8 -*-
"""CustomCNN2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17gmD7alNhFL8OsLS-AdxvTMgP-5wIrXd
"""



"""# Training CNN and saving the influence of every training example(for the test image with test_idx)

handling imports
"""


import numpy as np


import math
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

sns.set(color_codes=True)

import tensorflow as tf

from influence.all_CNN_c import All_CNN_C
from scripts.load_mnist import load_small_mnist, load_mnist

"""loading the dataset"""

import h5py
path_to_matrices = "data/default_labeled_balanced.hdf5"
dataset = h5py.File(path_to_matrices, 'r')

matrices = np.array(dataset['dense_matrices'])
labels = np.array(dataset['label_vectors'])

randomize = np.arange(len(matrices))
np.random.shuffle(randomize)
matrices = matrices[randomize]
labels = labels[randomize]

#from one hot to integer coding
i = 0
new_labels = []
for label in labels:
  index = np.argmax(label)
  new_labels.append(index)
  i+=1

labels = new_labels
labels = np.asarray(labels)

training_test_split = 0.8
index = int(training_test_split * len(matrices))

train_matrices = np.expand_dims(matrices[:index], axis=3)
train_labels = labels[:index]

validation_matrices = np.expand_dims(matrices[index + 1:], axis=3)
validation_labels = labels[index + 1:]

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet

train = DataSet(train_matrices, train_labels)
validation = DataSet(validation_matrices, validation_labels)
test = DataSet(validation_matrices, validation_labels)

data_sets = base.Datasets(train=train, validation=validation, test=test)

data_sets2 = load_small_mnist('data')

print(data_sets.train.labels.shape)
#print(data_sets2.train.labels.shape)
print(data_sets.train.x.shape)
#print(data_sets2.train.x.shape)

"""defining the CNN

training the CNN
"""

num_classes = 4
input_side = 128
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 1

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
hidden1_units = 8
hidden2_units = 8
hidden3_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]


model = All_CNN_C(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output', 
    log_dir='log',
    model_name='mnist_small_all_cnn_c')

num_steps = 5000
'''model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000,
    iter_to_switch_to_sgd=10000)
iter_to_load = num_steps - 1'''

"""calculating the influence"""

test_idx = 50

CNN_predicted_loss_diffs = model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(model.data_sets.train.labels)),
    force_refresh=True)
print("x")
"""saving the influence"""

np.savez(
    'output/CNN_results', 
    test_idx=test_idx,
    CNN_predicted_loss_diffs=CNN_predicted_loss_diffs
    
)
print("y")
"""# **Loading influences and plotting the most influential pictures**

getting Xtrain of the Training Dataset
"""

X_train = data_sets.train.x

"""loading the influences"""

f = np.load('output/CNN_results.npz')
    
test_idx = f['test_idx']
CNN_predicted_loss_diffs = f['CNN_predicted_loss_diffs']

print("Test Image:")
plt.spy(np.reshape(X_train[test_idx, :], [128, 128]))
print("Top 5 most influenatial matrices")

#x_train = []
#for counter, train_idx in enumerate(np.argsort(CNN_predicted_loss_diffs)[-5:]):
#    x_train.append(X_train[train_idx, :])
    

x_train = []
for counter, train_idx in enumerate(np.argsort(CNN_predicted_loss_diffs)[0:5]):
  x_train.append(X_train[train_idx, :])

plt.spy(np.reshape(x_train[0], [128, 128]))

plt.spy(np.reshape(x_train[1], [128, 128]))

plt.spy(np.reshape(x_train[2], [128, 128]))

plt.spy(np.reshape(x_train[3], [128, 128]))

plt.spy(np.reshape(x_train[4], [128, 128]))