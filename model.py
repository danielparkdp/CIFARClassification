from __future__ import absolute_import
from preprocess import get_data
from convolution import conv2d
import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()

		#hyperparameters
		self.batch_size = 64
		self.num_classes = 2
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
		self.epochs = 10



		#trainable parameters
		#for each layer there's a conv. filter and a conv. bias
		self.conv_1_filter = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], stddev=0.1, dtype=tf.float32))
		self.conv_1_bias = tf.Variable(tf.random.truncated_normal([16], stddev=0.1, dtype=tf.float32))

		self.conv_2_filter = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=0.1, dtype=tf.float32))
		self.conv_2_bias = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))

		self.conv_3_filter = tf.Variable(tf.random.truncated_normal([5, 5, 20, 20], stddev=0.1, dtype=tf.float32))
		self.conv_3_bias = tf.Variable(tf.random.truncated_normal([20], stddev=0.1, dtype=tf.float32))

		self.dense_w_1 = tf.Variable(tf.random.truncated_normal(shape=[320,64],stddev=0.1),dtype=tf.float32)
		self.dense_b_1 = tf.Variable(tf.random.truncated_normal(shape=[1,64],stddev=0.1),dtype=tf.float32)
		self.dense_w_2 = tf.Variable(tf.random.truncated_normal(shape=[64,32],stddev=0.1),dtype=tf.float32)
		self.dense_b_2 = tf.Variable(tf.random.truncated_normal(shape=[1,32],stddev=0.1),dtype=tf.float32)
		self.dense_w_3 = tf.Variable(tf.random.truncated_normal(shape=[32,self.num_classes],stddev=0.1),dtype=tf.float32)
		self.dense_b_3 = tf.Variable(tf.random.truncated_normal(shape=[1,self.num_classes],stddev=0.1),dtype=tf.float32)


	def call(self, inputs, is_testing=False):
		pass

	def loss(self, logits, labels):
		losses = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
		avg = tf.reduce_mean(losses)
		return avg

	def accuracy(self, logits, labels):
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
	pass

def test(model, test_inputs, test_labels):
	pass




def main():
	pass


if __name__ == '__main__':
	main()
