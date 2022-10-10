# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:34:03 2019

@author: Administrator
"""

from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints
import numpy as np
import tensorflow as tf
import quaternion

def QGF(sigma, theta, Lambda, psi, gamma):
    
        #sigma 带宽，取常数5
        #theta 不同的方向
        #lambda 不同的尺度
        #gamma 空间纵横比，一般取1
        #psi 相位，一般取0
     
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

        # ------这部分内容是为了确定卷积核的大小------
        # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    q= np.quaternion(0,1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3));
        # ------这部分内容是为了确定卷积核的大小------

        # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    qgb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.exp(q*(2 * np.pi / Lambda * x_theta + psi))
    qgb = quaternion.as_float_array(qgb)
    qgb = K.variable(value=qgb, dtype='float32', name='example_var')
    return qgb
          

class QuaternionGabor(Layer):
    
    def __init__(self,
                 theta, 
                 Lambda =7.0, 
                 sigma = 5.0, 
                 psi = 0.0, 
                 gamma = 1.0,  
                 **kwargs):
        
            #sigma 带宽，取常数5
            #theta 不同的方向
            #lambda 不同的尺度
            #gamma 空间纵横比，一般取1
            #psi 相位，一般取0
            
        super(QuaternionGabor, self).__init__(**kwargs)
        self.theta = initializers.Constant(value=30)
        self.Lambda = initializers.Constant(value=7)
        self.sigma = sigma
        self.psi = psi
        self.gamma = gamma
        
        
        
    def build(self, input_shape):
        
        ndim = len(input_shape)
 
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        param_shape = (input_shape[self.axis] // 4,)      
        
        self.theta = self.add_weight(shape=param_shape,
                                     name='theta',
                                     initializer=self.theta,
                                     trainable=True)
        
        self.Lambda = self.add_weight(shape=param_shape,
                                     name='Lambda',
                                     initializer=self.Lambda,
                                     trainable=True)
        
        super(QuaternionGabor, self).build(input_shape)
        
    def call(self, inputs):
        
        input_shape = K.int_shape(inputs)
        input_dim = input_shape[-1] // 4
        
        sig = self.sigma                     #sigma 带宽，取常数5
        gm = self.gamma                   #gamma 空间纵横比，一般取1
        ps = self.psi                    #psi 相位，一般取0
        th = self.theta*(np.pi)/180
        lm = self.Lambda
        kernel = QGF(sig,th,lm,ps,gm)
        
        q_r   = kernel[:, :, 0]
        q_i   = kernel[:, :, 1]
        q_j   = kernel[:, :, 2]
        q_k   = kernel[:, :, 3]
        
        convArgs = {"strides":       1,
                    "padding":       'valid',
                    "dilation_rate": 1}
        convFunc = {K.conv2d,
                    }
        
        cat_kernels_4_r = K.concatenate([q_r, -q_i, -q_j, -q_k], axis=-2)
        cat_kernels_4_i = K.concatenate([q_i, q_r, -q_k, q_j], axis=-2)
        cat_kernels_4_j = K.concatenate([q_j, q_k, q_r, -q_i], axis=-2)
        cat_kernels_4_k = K.concatenate([q_k, -q_j, q_i, q_r], axis=-2)
        cat_kernels_4_quaternion = K.concatenate([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], axis=-1)
        
        output = convFunc(inputs, cat_kernels_4_quaternion, **convArgs)
        output = inputs
        return output
    
    
    def get_config(self):
        config = {
             'theta': self.theta, 
             'Lambda': self.Lambda,
             'sigma': self.sigma,
             'psi': self.psi, 
             'gamma': self.gamma,
        }
        base_config = super(QuaternionGabor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        
        
        
        