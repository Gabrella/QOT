#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Chiheb Trabelsi

import numpy as np
from numpy.random import RandomState
from random import gauss
import keras.backend as K
from keras import initializers
from keras.initializers import Initializer
from keras.utils.generic_utils import (serialize_keras_object,
		deserialize_keras_object)


#####################################################################
#                   Quaternion Implementations                      #
#####################################################################
def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.
  Args:
    shape: Integer shape tuple or TF tensor shape.
  Returns:
    A tuple of scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

class qconv_cap_init(Initializer):
	# The standard complex initialization using
	# either the He or the Glorot criterion.
	def __init__(self, kernel_size, ch_i, n_i, ch_j, n_j,
			weight_dim, 
			criterion='he', seed=None):
		assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
		self.ch_i = ch_i
		self.n_i = n_i
		self.ch_j = ch_j
		self.n_j = n_j
		self.kernel_size = kernel_size
		self.weight_dim = weight_dim
		self.criterion = criterion
		self.seed = 1337 if seed is None else seed

	def __call__(self, shape, dtype=None):

		
		kernel_shape = tuple(self.kernel_size) + (self.ch_i, self.n_i, self.ch_j, self.n_j)
		

		fan_in, fan_out = _compute_fans(
				tuple(self.kernel_size) + (self.ch_i, self.n_i, self.ch_j, self.n_j)
				)

		# Quaternion operations start here

		if self.criterion == 'glorot':
			s = 1. / np.sqrt(2*(fan_in + fan_out))
		elif self.criterion == 'he':
			s = 1. / np.sqrt(2*fan_in)
		else:
			raise ValueError('Invalid criterion: ' + self.criterion)

		#Generating randoms and purely imaginary quaternions :
		number_of_weights = np.prod(kernel_shape) 
		v_i = np.random.uniform(0.0,1.0,number_of_weights)
		v_j = np.random.uniform(0.0,1.0,number_of_weights)
		v_k = np.random.uniform(0.0,1.0,number_of_weights)
		#Make these purely imaginary quaternions unitary
		for i in range(0, number_of_weights):
			norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
			v_i[i]/= norm
			v_j[i]/= norm
			v_k[i]/= norm
		v_i = v_i.reshape(kernel_shape)
		v_j = v_j.reshape(kernel_shape)
		v_k = v_k.reshape(kernel_shape)

		rng = RandomState(self.seed)
		modulus = rng.rayleigh(scale=s, size=kernel_shape)
		phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
		
		weight_r = modulus * np.cos(phase)
		weight_i = modulus * v_i*np.sin(phase)
		weight_j = modulus * v_j*np.sin(phase)
		weight_k = modulus * v_k*np.sin(phase)
		weight = np.concatenate([weight_r, weight_i, weight_j, weight_k], axis=-1)
		return weight


class qconv_init(Initializer):
	# The standard complex initialization using
	# either the He or the Glorot criterion.
	def __init__(self, kernel_size, input_dim,
			weight_dim, nb_filters=None,
			criterion='he', seed=None):

		# `weight_dim` is used as a parameter for sanity check
		# as we should not pass an integer as kernel_size when
		# the weight dimension is >= 2.
		# nb_filters == 0 if weights are not convolutional (matrix instead of filters)
		# then in such a case, weight_dim = 2.
		# (in case of 2D input):
		#     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
		# conv1D: len(kernel_size) == 1 and weight_dim == 1
		# conv2D: len(kernel_size) == 2 and weight_dim == 2
		# conv3d: len(kernel_size) == 3 and weight_dim == 3

		assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
		self.nb_filters = nb_filters
		self.kernel_size = kernel_size
		self.input_dim = input_dim
		self.weight_dim = weight_dim
		self.criterion = criterion
		self.seed = 1337 if seed is None else seed

	def __call__(self, shape, dtype=None):

		if self.nb_filters is not None:
			kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
		else:
			kernel_shape = (int(self.input_dim), self.kernel_size[-1])

		fan_in, fan_out = _compute_fans(
				tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
				)

		# Quaternion operations start here

		if self.criterion == 'glorot':
			s = 1. / np.sqrt(2*(fan_in + fan_out))
		elif self.criterion == 'he':
			s = 1. / np.sqrt(2*fan_in)
		else:
			raise ValueError('Invalid criterion: ' + self.criterion)

		#Generating randoms and purely imaginary quaternions :
		number_of_weights = np.prod(kernel_shape) 
		v_i = np.random.uniform(0.0,1.0,number_of_weights)
		v_j = np.random.uniform(0.0,1.0,number_of_weights)
		v_k = np.random.uniform(0.0,1.0,number_of_weights)
		#Make these purely imaginary quaternions unitary
		for i in range(0, number_of_weights):
			norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
			v_i[i]/= norm
			v_j[i]/= norm
			v_k[i]/= norm
		v_i = v_i.reshape(kernel_shape)
		v_j = v_j.reshape(kernel_shape)
		v_k = v_k.reshape(kernel_shape)

		rng = RandomState(self.seed)
		modulus = rng.rayleigh(scale=s, size=kernel_shape)
		phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
		
		weight_r = modulus * np.cos(phase)
		weight_i = modulus * v_i*np.sin(phase)
		weight_j = modulus * v_j*np.sin(phase)
		weight_k = modulus * v_k*np.sin(phase)
		weight = np.concatenate([weight_r, weight_i, weight_j, weight_k], axis=-1)

		return weight

class qdense_init(Initializer):
	# The standard complex initialization using
	# either the He or the Glorot criterion.
	def __init__(self, shape, criterion='he', seed=None):

		# `weight_dim` is used as a parameter for sanity check
		# as we should not pass an integer as kernel_size when
		# the weight dimension is >= 2.
		# nb_filters == 0 if weights are not convolutional (matrix instead of filters)
		# then in such a case, weight_dim = 2.
		# (in case of 2D input):
		#     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
		# conv1D: len(kernel_size) == 1 and weight_dim == 1
		# conv2D: len(kernel_size) == 2 and weight_dim == 2
		# conv3d: len(kernel_size) == 3 and weight_dim == 3

		self.shape = shape
		self.criterion = criterion
		self.seed = 1337 if seed is None else seed

	def __call__(self, shape, dtype=None):

		fan_in  = self.shape[0]
		fan_out = self.shape[1]

		# Quaternion operations start here

		if self.criterion == 'glorot':
			s = 1. / np.sqrt(2*(fan_in + fan_out))
		elif self.criterion == 'he':
			s = 1. / np.sqrt(2*fan_in)
		else:
			raise ValueError('Invalid criterion: ' + self.criterion)

		#Generating randoms and purely imaginary quaternions :
		number_of_weights = np.prod(self.shape) 
		v_i = np.random.uniform(0.0,1.0,number_of_weights)
		v_j = np.random.uniform(0.0,1.0,number_of_weights)
		v_k = np.random.uniform(0.0,1.0,number_of_weights)
		#Make these purely imaginary quaternions unitary
		for i in range(0, number_of_weights):
			norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
			v_i[i]/= norm
			v_j[i]/= norm
			v_k[i]/= norm
		v_i = v_i.reshape(self.shape)
		v_j = v_j.reshape(self.shape)
		v_k = v_k.reshape(self.shape)

		rng = RandomState(self.seed)
		modulus = rng.rayleigh(scale=s, size=self.shape)
		phase = rng.uniform(low=-np.pi, high=np.pi, size=self.shape)
		
		weight_r = modulus * np.cos(phase)
		weight_i = modulus * v_i*np.sin(phase)
		weight_j = modulus * v_j*np.sin(phase)
		weight_k = modulus * v_k*np.sin(phase)

		weight = np.concatenate([weight_r, weight_i, weight_j, weight_k], axis=-1)
		
		return weight

class sqrt_init(Initializer):
	def __call__(self, shape, dtype=None):
		return K.constant(1 / K.sqrt(2), shape=shape, dtype=dtype)


