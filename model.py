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
		conv1_layer = tf.nn.conv2d(inputs, self.conv_1_filter, [1,2,2,1], "SAME")
		conv1_out = tf.nn.bias_add(conv1_layer, self.conv_1_bias)
		m, v = tf.nn.moments(conv1_out, [0])
		conv1_norm = tf.nn.batch_normalization(conv1_out, m, v, None, None, 0.001)
		conv1 = tf.nn.relu(conv1_norm)

		max_pool1 = tf.nn.max_pool(conv1, [3,3], [1,2,2,1], "SAME")

		conv2_layer = tf.nn.conv2d(max_pool1, self.conv_2_filter, [1,1,1,1], "SAME")
		conv2_out = tf.nn.bias_add(conv2_layer, self.conv_2_bias)
		m, v = tf.nn.moments(conv2_out, [0])
		conv2_norm = tf.nn.batch_normalization(conv2_out, m, v, None, None, 0.001)
		conv2 = tf.nn.relu(conv2_norm)

		max_pool2 = tf.nn.max_pool(conv2, [2,2], [1,2,2,1], "SAME")

		if is_testing:
			conv3_layer = conv2d(max_pool2, self.conv_3_filter, [1,1,1,1], "SAME")
		else:
			conv3_layer = tf.nn.conv2d(max_pool2, self.conv_3_filter, [1,1,1,1], "SAME")
		conv3_out = tf.nn.bias_add(conv3_layer, self.conv_3_bias)
		m, v = tf.nn.moments(conv3_out, [0])
		conv3_norm = tf.nn.batch_normalization(conv3_out, m, v, None, None, 0.001)
		conv3 = tf.nn.relu(conv3_norm)

		input_size = conv3.get_shape().as_list()
		size2 = input_size[-1]*input_size[-2]*input_size[-3]
		result = tf.reshape(conv3, [-1, size2])

		result1 = tf.matmul(result, self.dense_w_1)+self.dense_b_1
		result1 = tf.nn.dropout(result1, 0.3)
		result2 = tf.matmul(result1, self.dense_w_2)+self.dense_b_2
		result2 = tf.nn.dropout(result2, 0.3)
		result3 = tf.matmul(result2, self.dense_w_3)+self.dense_b_3
		return result3

	def loss(self, logits, labels):
		losses = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
		avg = tf.reduce_mean(losses)
		return avg

	def accuracy(self, logits, labels):
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
	num_rows = len(train_inputs)
	r = range(0,num_rows)
	r = tf.random.shuffle(r)
	inp = tf.gather(train_inputs, r)
	lab = tf.gather(train_labels, r)
	i = model.batch_size

	while i <= num_rows:
		curr = inp[(i-model.batch_size):i]
		currlab = lab[(i-model.batch_size):i]
		curr = tf.image.random_flip_left_right(curr)
		with tf.GradientTape() as tape:
			log = model.call(curr)
			loss = model.loss(log, currlab)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		i+=model.batch_size

def test(model, test_inputs, test_labels):
	log = model.call(test_inputs, is_testing=True)
	acc = model.accuracy(log, test_labels)
	return acc




def main():
	# store data in folder called data
	train_inp, train_lab = get_data("data/train", 3, 5)
	test_inp, test_lab = get_data("data/test", 3, 5)
	mod = Model()
	for i in range(0,mod.epochs):
		train(mod, train_inp, train_lab)
	acc = test(mod, test_inp, test_lab)
	print(acc)
	return


if __name__ == '__main__':
	main()
