# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

examples, labels, test_examples, test_labels = mnist.load_data(one_hot=True)
examples = examples.reshape([-1, 28, 28, 1])
test_examples = test_examples.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(
    network,
    optimizer='adam',
    learning_rate=0.01,
    loss='categorical_crossentropy',
    name='target'
)

# Training
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(
    examples,
    labels,
    n_epoch=20,
    validation_set=(test_examples, test_labels),
    snapshot_step=100,
    show_metric=True,
)
