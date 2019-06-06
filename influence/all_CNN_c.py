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

import os.path
import time
import IPython
import tensorflow as tf
import math

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
        model = Sequential()
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

        model.add(Dense(10))
        model.add(Activation("softmax"))

        optimizer = optimizers.SGD(lr=0.0001)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer, metrics=['accuracy'])

        return model



    def get_initializers_of_keras2(self,model):
        #get the weights of each layer and flatten them
        weights1 = model.layers[0].get_weights()[0]
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


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        model2 = self.test_keras_model2()
        initializers = self.get_initializers_of_keras2(model2)

        weights = tf.get_variable(
            'weights',
            initializer= initializers [0],
            dtype = tf.float32)
        biases = tf.get_variable(
            'biases',
            shape=[32],
            dtype=tf.float32)
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.relu(conv2d(input_x, weights_reshaped, stride) + biases)
        return hidden


    def get_all_params(self):


        all_params = []
        for layer in ['conv1', 'dense1','dense2','dense3']:
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


    def inference(self, input_x):
        model2 = self.test_keras_model2()
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
        conv1_reshaped = tf.reshape(conv1, [-1, 508032])
        #conv1_reshaped = tf.reshape(conv1,[-1])
        #print(conv1.shape)
        #print(tf.reshape(conv1,[-1]).shape)
        # conv1_reshaped= tf.reduce_mean(conv1, axis=[1, 2])
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
            print("conv1 reshape")
            print(conv1_reshaped.shape)
            print("bias 1")
            print(biases1.shape)
            dense1 = tf.matmul(conv1_reshaped,tf.reshape(weights1, [508032, 128]) ) + biases1
            print("dense 1")
            print(dense1.shape)

        # second dense layer
        with tf.variable_scope('dense2'):
            weights2 = tf.get_variable(
                'weights',
                initializer=initializers[4])
            biases2 = tf.get_variable(
                'biases',
                initializer=initializers[5])
            print("bias 2")
            print(biases2.shape)
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

        return dense3

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds