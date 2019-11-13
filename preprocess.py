import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

def unpickle(file):
	"""
	prepare data based on data's "pickled" state when given to us
	:param file: the file with data
	:return: dictionary of data
	"""
	pass


def get_data(file_path, first_class, second_class):
	"""
	Since we are just comparing 2 classes for now, identify which classes to process and pass in file path
	:param file_path: file path for inputs and labels (our CIFAR)
	:param first_class:  an integer (0-9) representing the first target
	:param first_class:  an integer (0-9) representing the second target
	:return: inputs and tensor of the given labels
	"""
	pass
