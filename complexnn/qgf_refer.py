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
    return qgb
          
def convolve(img,fil,mode = 'same'):                #分别提取三个通道
    
    if mode == 'fill':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w),(0, 0)), 'constant')
    conv_img = _convolve(img,fil)              #然后去进行卷积操作
    return conv_img                                   #返回卷积后的结果
    
def _convolve(img,fil):         
        
    fil_heigh = fil.shape[0]                        #获取卷积核(滤波)的高度
    fil_width = fil.shape[1]                        #获取卷积核(滤波)的宽度
        
    conv_heigh = img.shape[0] - fil.shape[0] + 1    #确定卷积结果的大小
    conv_width = img.shape[1] - fil.shape[1] + 1
    
    conv = []
    for b in range(conv_heigh*conv_width):
        conv.append(np.quaternion(b,0,0,0))
    conv = np.array(conv).reshape(conv_heigh,conv_width)
        
    for i in range(conv_heigh):
        for j in range(conv_width):                 #逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh,j:j + fil_width ],fil)
    return conv
        
def wise_element_sum(img,fil):
    res = (img * fil).sum() 
    return res          

class QuaternionGabor(Layer):
    
    def __init__(self,
                 theta, 
                 Lambda =7, 
                 sigma = 5, 
                 psi = 0, 
                 gamma = 1,  
                 **kwargs):
            #sigma 带宽，取常数5
            #theta 不同的方向
            #lambda 不同的尺度
            #gamma 空间纵横比，一般取1
            #psi 相位，一般取0
        self.sigma = sigma
        self.theta = initializers.Constant(value=30)
        self.Lambda = initializers.Constant(value=7)
        self.psi = psi
        self.gamma = gamma
        
        super(QuaternionGabor, self).__init__(**kwargs)
        
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
        
        super(QuaternionGabor, self).build(input_shape)  # 一定要在最后调用它
        
    def call(self, inputs):
        
        input_shape = K.int_shape(inputs)
        #input_dim = input_shape[-1] // 4
        
        inputs2 = K.eval(inputs)
        
        i = inputs2[:,:,1]
        j = inputs2[:,:,2]
        k = inputs2[:,:,3]
        
        list = []
        for a in range(input_shape[0]*input_shape[1]):
            list.append(np.quaternion(a,0,0,0))
        Q = np.array(list).reshape(input_shape[0],input_shape[1])
        
        for m in range(input_shape[0]):
            for n in range(input_shape[1]):
                Q[m,n] = np.quaternion(0,i[m,n],j[m,n],k[m,n])
        
        #将其变回矩阵形势
        #Qarray = quaternion.as_float_array(Q)  
        sig = self.sigma                     #sigma 带宽，取常数5
        gm = self.gamma                   #gamma 空间纵横比，一般取1
        ps = self.psi                    #psi 相位，一般取0
        th = self.theta*(np.pi)/180
        lm = self.Lambda
        kernel = QGF(sig,th,lm,ps,gm)
        dest = convolve(Q,kernel,mode = 'same')
        destarray = quaternion.as_float_array(dest)
        output = 
        
        return output
    
    def get_config(self):
        config = {
             'sigma': self.sigma,
             'theta': self.theta, 
             'Lambda': self.Lambda,
             'psi': self.psi, 
             'gamma': self.gamma,    
                
        }
        base_config = super(QuaternionGabor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        
        
        
        