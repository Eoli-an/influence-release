from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
import os
from keras.regularizers import l2
import os.path
import time
import IPython
import tensorflow as tf
import math
import tensorflow.math

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras
from keras.models import load_model

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense, MaxPooling2D
from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras
from keras.utils import multi_gpu_model
from time import time

import os
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
import os
from os.path import abspath
import h5py

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet

def conv2d(x, W, r):
    return tf.nn.conv2d(x, W, strides=[1, r, r, 1], padding='VALID')

def softplus(x):
    return tf.log(tf.exp(x) + 1)


class All_CNN_C(GenericNeuralNet):

    def __init__(self, input_side, input_channels, conv_patch_size, hidden1_units, hidden2_units, hidden3_units, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        self.input_side = input_side
        self.input_channels = input_channels
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.conv_patch_size = conv_patch_size
        self.hidden1_units = hidden1_units
        self.hidden2_units = hidden2_units
        self.hidden3_units = hidden3_units

        super(All_CNN_C, self).__init__(**kwargs)



    def test_keras_model2(self):
        '''model = Sequential()
        output_size = 32
        kernel_size = 3
        new_output_size = 0

        model.add(Conv2D(output_size, (kernel_size, kernel_size), input_shape=(128, 128, 1)))
        model.add(Activation("relu"))
        new_output_size = output_size

        model.add(Flatten())

        model.add(Dense(new_output_size * 4))
        model.add(Activation("relu"))
        for i in range(1, 2):
            model.add(Dense(int((new_output_size * 4) / i)))
            model.add(Activation("relu"))

        model.add(Dense(4))
        model.add(Activation("softmax"))

        optimizer = optimizers.SGD(lr=0.0001)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer, metrics=['accuracy'])

        return model'''
        __neuron_list = [82, 253, 85, 60, 19, 32, 19]

        l2reg = l2(0.04312)
        model = Sequential()
        output_size = 32
        kernel_size = 3
        layer_counter = 0

        if 4 != 0:
            model.add(Conv2D(__neuron_list[layer_counter], (kernel_size, kernel_size), input_shape=(128, 128, 1)))
            model.add(Activation("tanh"))
            layer_counter += 1

            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

        for i in range(1, 4):
            model.add(Conv2D(__neuron_list[layer_counter], (kernel_size, kernel_size)))
            model.add(Activation("tanh"))
            layer_counter += 1

            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

        model.add(Flatten())
        model.add(Dropout(0.1))

        model.add(Dense(4))
        model.add(Activation("sigmoid"))

        optimizer = optimizers.SGD(0.04312)

        model.compile(loss="mean_squared_error",
                      optimizer=optimizer, metrics=['accuracy'])

        return model



    def get_initializers_of_keras2(self,model):
        #get the weights of each layer and flatten them
        '''weights1 = model.layers[0].get_weights()[0]
        weights1 = weights1.flatten()
        biases1 = model.layers[0].get_weights()[1]

        weights2 = model.layers[3].get_weights()[0]
        weights2 = weights2.flatten()
        biases2 = model.layers[3].get_weights()[1]

        weights3 = model.layers[5].get_weights()[0]
        weights3 = weights3.flatten()
        biases3 = model.layers[5].get_weights()[1]

        weights4 = model.layers[7].get_weights()[0]
        weights4 = weights4.flatten()
        biases4 = model.layers[7].get_weights()[1]

        return [weights1,biases1,weights2,biases2,weights3,biases3,weights4,biases4]


        weights1 = model.layers[0].get_weights()[0]
        weights1 = weights1.flatten()
        biases1 = model.layers[0].get_weights()[1]


        weights4 = model.layers[3].get_weights()[0]
        weights4 = weights4.flatten()
        biases4 = model.layers[3].get_weights()[1]


        weights7 = model.layers[6].get_weights()[0]
        weights7 = weights7.flatten()
        biases7 = model.layers[6].get_weights()[1]


        weights10 = model.layers[9].get_weights()[0]
        weights10 = weights10.flatten()
        biases10 = model.layers[9].get_weights()[1]


        weights15 = model.layers[14].get_weights()[0]
        weights15 = weights15.flatten()
        biases15 = model.layers[14].get_weights()[1]

        return [weights1,biases1,weights4,biases4,weights7,biases7,weights10,biases10,weights15,biases15]
        '''

        params = model.get_weights()
        for i,par in enumerate(params):
            params[i] = par.flatten()

        return params


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, index,stride):
        #model3 = self.test_keras_model2()
        model2 = load_model(os.path.join("data","final_net.hdf5"))



        initializers = self.get_initializers_of_keras2(model2)

        weights = tf.get_variable(
            'weights',
            initializer= initializers [index],
            dtype = tf.float32)
        biases = tf.get_variable(
            'biases',
            initializer=initializers[index + 1],
            dtype=tf.float32)
        #print(weights.shape)
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.tanh(conv2d(input_x, weights_reshaped, stride) + biases)
        return hidden


    def get_all_params(self):

        all_params = []
        for layer in ['conv1', 'conv2','conv3','conv4','dense1']:
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)
        return all_params        
        

    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


    def inference(self, input_x):

        #model2 = keras.models.load_model("data/final_net.h5")
        model2 = self.test_keras_model2()
        initializers = self.get_initializers_of_keras2(model2)
        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        last_layer_units = 128

        with tf.variable_scope('conv1'):
            # variablen definieren und conv und relu alles in einem schritt
            conv1 = self.conv2d_softplus(input_reshaped, self.conv_patch_size, self.input_channels, 82, 0,stride=1)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv2'):
            # variablen definieren und conv und relu alles in einem schritt
            conv2 = self.conv2d_softplus(conv1, self.conv_patch_size, 82, 253,2, stride=1)

            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope('conv3'):
            # variablen definieren und conv und relu alles in einem schritt
            conv3 = self.conv2d_softplus(conv2, self.conv_patch_size, 253, 85,4, stride=1)
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv4'):
            # variablen definieren und conv und relu alles in einem schritt
            conv4 = self.conv2d_softplus(conv3, self.conv_patch_size, 85, 60,6, stride=1)
            conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv_reshaped = tf.reshape(conv4, [-1, 2160])

        with tf.variable_scope('dense1'):
            # definition der weights
            weights1 = tf.get_variable(
                'weights',
                initializer=initializers[8])
            # def der biases
            biases1 = tf.get_variable(
                'biases',
                initializer=initializers[9])
            dense1 = tf.matmul(conv_reshaped, tf.reshape(weights1, [ 2160,
                                                                     self.num_classes])) + biases1

        return tf.math.sigmoid(dense1)
        '''model2 = self.test_keras_model2()
        initializers = self.get_initializers_of_keras2(model2)

        #input, which is a vector gets reshaped to an image
        # -1:amount of pictures, input side:lenght/width channesl:black n white
        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        last_layer_units = 128
        # first and only convolutional layer
        with tf.variable_scope('conv1'):
            # variablen definieren und conv und relu alles in einem schritt
            conv1 = self.conv2d_softplus(input_reshaped, self.conv_patch_size, self.input_channels, 32, stride=1)

        # jetzt muss reshaped werden damit das dense layer verbunden werden kann
        # groesse des outputs des conv layers passt sich dem input an(groesse des bildes)
        conv1_reshaped = tf.reshape(conv1, [-1, (self.input_side-2)*(self.input_side-2)*32])

        # first dense layer
        with tf.variable_scope('dense1'):
            # definition der weights
            weights1 = tf.get_variable(
                'weights',
                initializer=initializers[2])
            # def der biases
            biases1 = tf.get_variable(
                'biases',
                initializer=initializers[3])
            dense1 = tf.matmul(conv1_reshaped,tf.reshape(weights1, [(self.input_side-2)*(self.input_side-2)*32, 128]) ) + biases1


        # second dense layer
        with tf.variable_scope('dense2'):
            weights2 = tf.get_variable(
                'weights',
                initializer=initializers[4])
            biases2 = tf.get_variable(
                'biases',
                initializer=initializers[5])
            dense2 = tf.matmul(dense1, tf.reshape(weights2, [128, 128])) + biases2

        # last dense layer, output layer

        with tf.variable_scope('dense3'):
            weights3 = tf.get_variable(
                'weights',
                initializer=initializers[6])
            biases3 = tf.get_variable(
                 'biases',
                initializer=initializers[7])

            dense3 = tf.matmul(dense2, tf.reshape(weights3, [last_layer_units, self.num_classes])) + biases3

        return dense3'''


    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds