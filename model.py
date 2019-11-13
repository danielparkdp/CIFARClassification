from __future__ import absolute_import
from preprocess import get_data
from convolution import conv2d
import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
	def __init__(self):
		pass


	def call(self, inputs, is_testing=False):
		pass

	def loss(self, logits, labels):
		pass

	def accuracy(self, logits, labels):
		pass


def train(model, train_inputs, train_labels):
	pass

def test(model, test_inputs, test_labels):
	pass




def main():
	pass


if __name__ == '__main__':
	main()
