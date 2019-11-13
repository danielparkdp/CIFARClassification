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
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Since we are just comparing 2 classes for now, identify which classes to process and pass in file path
	:param file_path: file path for inputs and labels (our CIFAR)
	:param first_class:  an integer (0-9) representing the first target
	:param first_class:  an integer (0-9) representing the second target
	:return: inputs and tensor of the given labels
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']

	inpl = []
	labl = []

	for i in range(0,len(inputs)):
		if labels[i] == first_class:
			inpl.append(inputs[i])
			labl.append(0)
		elif labels[i] == second_class:
			inpl.append(inputs[i])
			labl.append(1)

	inp = np.array(inpl)
	lab = np.array(labl)
	inp = np.reshape(inp, (-1, 3, 32 ,32))
	inp = inp.transpose((0,2,3,1))
	lab = tf.one_hot(lab, depth=2)
	inp = tf.cast(inp, tf.float32)
	return inp, lab
