#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from collections import namedtuple
from tensorflow.examples.tutorials.mnist import input_data


Layer = namedtuple('Layer', 'weights, biases')
Data = namedtuple('Data', 'examples, labels')


class LeNet5(object):
    def __init__(self, graph=None):
        # The graph object used for this model
        self.graph = graph or tf.Graph()

        # Define some handles to nodes in the graph
        # For feed nodes
        self.examples = None
        self.labels = None

        # For training
        self.optimize = None
        self.prediction = None

        # For predictions (inference)
        self.accuracy_tensor = None
        self.accuracy = None
        self.loss_tensor = None
        self.loss = None

        # Setup the graph
        self._setup_model()

    def predict(self, session, image):
        prediction = session.run(self.prediction, feed_dict={
            self.examples: [image]
        })

        return prediction

    def train(self, session, data, compute_accuracy=False):
        fetches = [self.optimize]
        if compute_accuracy:
            fetches.extend([self.accuracy_tensor, self.loss_tensor])

        result = session.run(fetches, feed_dict={
            self.examples: data.examples,
            self.labels: data.labels
        })

        if compute_accuracy:
            self.accuracy, self.loss = result[1:]

    def test(self, session, data):
        self.train(session, data, compute_accuracy=True)
        return self.accuracy

    def _setup_model(self):
        with self.graph.as_default():
            # Input data
            self.examples = tf.placeholder(tf.float32, shape=[None, 784])
            self.labels = tf.placeholder(tf.float32, shape=[None, 10])

            # Convolutional Layers
            first_layer = self._create_layer(5, 5, 1, 32)
            second_layer = self._create_layer(5, 5, 32, 64)

            # Reshape pixel vectors into image volumes
            examples = tf.reshape(self.examples, [-1, 28, 28, 1])

            # Propagate examples through first layer
            first_output = self._convolve(examples, first_layer)
            first_output = self._pool(first_output)

            # Propagate first layer output through second layer
            second_output = self._convolve(first_output, second_layer)
            second_output = self._pool(second_output)

            # Fully Connected Layers
            third_layer = self._create_layer(7 * 7 * 64, 1024)
            fourth_layer = self._create_layer(1024, 10)

            # Flatten the image volumes into pixel vectors
            flattened = tf.reshape(second_output, [-1, 7 * 7 * 64])

            # Propagate through first fully connected layer
            third_output = self._propagate(flattened, third_layer)

            # Apply dropout for regularization
            third_output = tf.nn.dropout(third_output, keep_prob=0.5)

            # Propagate through final layer to compute class scores
            scores = self._propagate(third_output, fourth_layer, None)
            probabilities = tf.nn.softmax(scores)

            # Pick the highest likelyhood as the predicted class
            self.predictions = tf.argmax(probabilities, dimension=1)

            # Compute loss value
            self.loss_tensor = self._cross_entropy(self.labels, probabilities)

            # Adam optimizer chosen to minimize the loss
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.optimize = optimizer.minimize(self.loss_tensor)

            # Find true positives and true negatives
            hits = tf.equal(
                self.predictions,
                tf.argmax(self.labels, dimension=1),
            )

            # Compute the accuracy
            self.accuracy_tensor = tf.reduce_mean(tf.cast(hits, tf.float32))

    @staticmethod
    def _create_layer(*shape):
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=shape[-1:]))

        return Layer(weights, bias)

    @staticmethod
    def _convolve(input_volume, conv_layer):
        output_volume = tf.nn.conv2d(
            input_volume,
            conv_layer.weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        return tf.nn.relu(output_volume + conv_layer.biases)

    @staticmethod
    def _pool(input_volume):
        return tf.nn.max_pool(
            input_volume,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

    @staticmethod
    def _propagate(input_features, layer, activation=tf.nn.relu):
        scores = tf.matmul(input_features, layer.weights) + layer.biases
        return activation(scores) if activation else scores

    @staticmethod
    def _cross_entropy(p, q):
        return tf.reduce_mean(-tf.reduce_sum(p * tf.log(q), [1]))

    def __repr__(self):
        return 'Accuracy: {0} | Loss: {1}'.format(self.accuracy, self.loss)


def test(session, net, mnist):
    print('Testing!')
    data = Data(mnist.test.images, mnist.test.labels)
    print(net.test(session, data))


def train(session, net, batch, epoch, checkpoint):
    if epoch % checkpoint == 0:
        net.train(session, Data(*batch), compute_accuracy=True)
        print('Epoch {0} | {1}'.format(int(epoch / checkpoint), net))
    else:
        net.train(session, Data(*batch), compute_accuracy=False)


def main():
    net = LeNet5()
    number_of_epochs = 10000
    checkpoint = number_of_epochs / 100
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Session(graph=net.graph) as session:
        tf.initialize_all_variables().run()
        for epoch in range(1, number_of_epochs):
            batch = mnist.train.next_batch(100)
            train(session, net, batch, epoch, checkpoint)
        test(session, net, mnist)

if __name__ == '__main__':
    main()
