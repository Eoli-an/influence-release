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


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        weights = tf.get_variable(
            'weights', 
            [conv_patch_size * conv_patch_size * input_channels * output_channels],
            initializer= tf.constant_initializer(0.0),
            dtype = tf.float32)
        biases = variable(
            'biases',
            [output_channels],
            tf.constant_initializer(0.0))
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.relu(conv2d(input_x, weights_reshaped, stride) + biases)
        print(weights.shape)
        print(weights_reshaped.shape)
        print(hidden.shape)
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

    def test_keras_model(self):
        a = 5
        return a

    def get_initializers_of_keras(self):
        b = 6
        return b

    def inference(self, input_x):


        #input, which is a vector gets reshaped to an image
        # -1:amount of pictures, input side:lenght/width channesl:black n white
        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        last_layer_units = 128
        # first and only convolutional layer
        with tf.variable_scope('conv1'):
            # variablen definieren und conv und relu alles in einem schritt
            conv1 = self.conv2d_softplus(input_reshaped, self.conv_patch_size, self.input_channels, 32, stride=1)
            

        # jetzt muss reshaped werden damit das dense layer verbunden werden kann
        conv1_reshaped = tf.reshape(conv1,[-1,21632])
        #conv1_reshaped= tf.reduce_mean(conv1, axis=[1, 2])
        # first dense layer
        with tf.variable_scope('dense1'):
            # definition der weights
            weights1 = variable(
                'weights',
                [21632*128],
                tf.truncated_normal_initializer(stddev=0.2))
            #def der biases
            biases1 = variable(
                'biases',
                [128],
                tf.constant_initializer(0.0))
            dense1 = tf.matmul(conv1_reshaped, tf.reshape(weights1,[21632,128])) + biases1
            
        # second dense layer
        with tf.variable_scope('dense2'):
            weights2 = variable(
                'weights',
                [128*128],
                tf.truncated_normal_initializer(stddev=0.2))
            biases2 = variable(
                'biases',
                [128],
                tf.constant_initializer(0.0))
            dense2 = tf.matmul(dense1, tf.reshape(weights2,[128,128])) + biases2

        # last dense layer, output layer

        with tf.variable_scope('dense3'):

            weights3 = variable(
                'weights', 
                [last_layer_units* self.num_classes],
                tf.truncated_normal_initializer(stddev=0.2))
            biases3 = variable(
                'biases',
                [self.num_classes],
                tf.constant_initializer(0.0))

            dense3 = tf.matmul(dense2, tf.reshape(weights3,[last_layer_units,self.num_classes])) + biases3
        return dense3

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds