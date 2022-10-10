# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:33:26 2020

@author: zhouyu
"""
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Lambda, Layer, InputSpec, Convolution1D, Convolution2D, add, multiply, Activation, Input, concatenate
from keras.layers.convolutional import _Conv
from keras.layers.merge import _Merge
from keras.layers.recurrent import Recurrent
from keras.utils import conv_utils
from keras.models import Model
import numpy as np
from .init import *
import sys
from keras.utils.conv_utils import normalize_data_format

class QConv2DTranspose(Layer):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.

    # References
        - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 normalize_weight=False,
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 gamma_diag_initializer=sqrt_init,
                 gamma_off_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 init_criterion='he',
                 seed=None,
                 spectral_parametrization=False,
                 epsilon=1e-7,
                 **kwargs):
        super(QConv2DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.normalize_weight = normalize_weight
        self.init_criterion = init_criterion
        self.spectral_parametrization = spectral_parametrization
        self.epsilon = epsilon
        self.kernel_initializer = sanitizedInitGet(kernel_initializer)
        self.bias_initializer = sanitizedInitGet(bias_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=4)
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis] // 4
        self.kernel_shape = self.kernel_size + (self.filters, input_dim)
    
        kls = {'quaternion': qconv_init}[self.kernel_initializer]
        kern_init = kls(
            kernel_size=self.kernel_size,
            input_dim=input_dim,
            weight_dim=2,
            nb_filters=self.filters,
            criterion=self.init_criterion)
    
        self.kernel = self.add_weight(
            self.kernel_shape,
            initializer=kern_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
    
        if self.use_bias:
            bias_shape = (4 * self.filters,)
            self.bias = self.add_weight(
                bias_shape,
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        else:
            self.bias = None
            
        # Set input spec.
        self.input_spec = InputSpec(ndim= 4,
                                    axes={channel_axis: input_dim * 4})
        self.built = True
        
    def call(self, inputs):
        input_shape = K.shape(inputs)
        #input_dim    = input_shape[-1] // 4
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2
    
        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding)
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            
        f_r   = self.kernel[:, :, :self.filters, :]
        f_i   = self.kernel[:, :, self.filters:self.filters*2, :]
        f_j   = self.kernel[:, :, self.filters*2:self.filters*3, :]
        f_k   = self.kernel[:, :, self.filters*3:, :]
        
        cat_kernels_4_r = K.concatenate([f_r, -f_i, -f_j, -f_k], axis=-1)
        cat_kernels_4_i = K.concatenate([f_i, f_r, -f_k, f_j], axis=-1)
        cat_kernels_4_j = K.concatenate([f_j, f_k, f_r, -f_i], axis=-1)
        cat_kernels_4_k = K.concatenate([f_k, -f_j, f_i, f_r], axis=-1)
        cat_kernels_4_quaternion = K.concatenate([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], axis=-2)
    
        outputs = K.conv2d_transpose(
            inputs,
            cat_kernels_4_quaternion,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format)
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length(
            output_shape[h_axis], stride_h, kernel_h, self.padding)
        output_shape[w_axis] = conv_utils.deconv_length(
            output_shape[w_axis], stride_w, kernel_w, self.padding)
        return tuple(output_shape)

    def get_config(self):
        config = super(QConv2DTranspose, self).get_config()
        config.pop('dilation_rate')
        return config
    
    
def sanitizedInitGet(init):
    if   init in ["sqrt_init"]:
        return sqrt_init
    elif init in ["complex", "complex_independent",
                  "glorot_complex", "he_complex",
                  "quaternion", "quaternion_independent"]:
        return init
    else:
        return initializers.get(init)

def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    elif init == "quaternion" or isinstance(init, QuaternionInit):
        return "quaternion"
    elif init == "quaternion_independent" or isinstance(init, QuaternionIndependentFilters):
        return "quaternion_independent"
    else:
        return initializers.serialize(init)   
    