#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zglyy
"""
from wsgiref import validate
import tensorflow as tf
tf.random.set_seed(42)
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import numpy as np
np.random.seed(42)

# load orthogonal features
train_1=np.load('orthognal_npy/train_1_RAFDB_v1.npy',encoding = "latin1")
train_2=np.load('orthognal_npy/train_2_RAFDB_v1.npy',encoding = "latin1")
train_3=np.load('orthognal_npy/train_3_RAFDB_v1.npy',encoding = "latin1")
train_label=np.load('orthognal_npy/train_label_RAFDB_v1.npy',encoding = "latin1")
test_1=np.load('orthognal_npy/test_1_RAFDB_v1.npy',encoding = "latin1")
test_2=np.load('orthognal_npy/test_2_RAFDB_v1.npy',encoding = "latin1")
test_3=np.load('orthognal_npy/test_3_RAFDB_v1.npy',encoding = "latin1")
test_label=np.load('orthognal_npy/test_label_RAFDB_v1.npy',encoding = "latin1")

# average the three sub-features and put them into a quaternion matrix
q_train=np.zeros([train_1.shape[0],train_1.shape[1],train_1.shape[2],train_1.shape[-1]*4])
train_r=(train_1+train_2+train_3)/3
q_train[:,:,:,:train_1.shape[-1]]=train_r
q_train[:,:,:,train_1.shape[-1]:2*train_1.shape[-1]]=train_1
q_train[:,:,:,2*train_1.shape[-1]:3*train_1.shape[-1]]=train_2
q_train[:,:,:,3*train_1.shape[-1]:]=train_3
train = np.transpose(q_train,(0,3,1,2))
train = np.reshape(train,(train_1.shape[0],256*4,49))

q_test=np.zeros([test_1.shape[0],test_1.shape[1],test_1.shape[2],test_1.shape[-1]*4])
test_r=(test_1+test_2+test_3)/3
q_test[:,:,:,:test_1.shape[-1]]=test_r
q_test[:,:,:,test_1.shape[-1]:2*test_1.shape[-1]]=test_1
q_test[:,:,:,2*test_1.shape[-1]:3*test_1.shape[-1]]=test_2
q_test[:,:,:,3*test_1.shape[-1]:]=test_3
test = np.transpose(q_test,(0,3,1,2))
test = np.reshape(test,(test_1.shape[0],256*4,49))


input_shape = (256*4, 49)
num_classes = 7
learning_rate = 0.00001
weight_decay = 0.0001
batch_size = 8
num_epochs = 400
num_patches = 256*4
projection_dim = 48
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024]


from   complexnn      import *
from tensorflow.keras.layers import (
    Dense,   
)
# Q-MHSA module
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = QuaternionDense(embed_dim)
        self.key_dense = QuaternionDense(embed_dim)
        self.value_dense = QuaternionDense(embed_dim)
        self.combine_heads = QuaternionDense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


def QF_Net(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Activation(tf.nn.gelu)(x)
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
    return x

def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionDense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        # 降为全连接
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        # encoded = patch + self.position_embedding(positions)
        return encoded
    

def create_qvit_classifier():
    inputs = layers.Input(shape=input_shape)

    # position embedding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(inputs)
    
    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        attention_output = MultiHeadSelfAttention(projection_dim, num_heads)(x1)
        
        x2 = layers.Add()([attention_output, encoded_patches])
   
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x4 = tf.keras.layers.Reshape((32,32,48))(x3)
       
        x5 = QF_Net(x4, hidden_units=transformer_units, dropout_rate=0.3)
        
        x6 = tf.keras.layers.Reshape((256*4, 48))(x5)
      
        encoded_patches = layers.Add()([x6, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "./tmp/RAFDB/model_{epoch:03d}-{val_accuracy:.4f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=train,
        y=train_label,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(test, test_label),
        callbacks=[checkpoint_callback],
    )

    return history

vit_classifier = create_qvit_classifier()
history = run_experiment(vit_classifier)

'''evaluate'''
# checkpoint_filepath="./tmp/RAFDB/model-0.8997.h5"
# model = create_qvit_classifier()
# # 重新创建完全相同的模型
# model.load_weights(checkpoint_filepath)
# # 加载后重新编译模型，否则您将失去优化器的状态
# optimizer = tf.optimizers.Adam(
#         learning_rate=learning_rate#, weight_decay=weight_decay
#     )
# model.compile(
#     optimizer=optimizer,
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[
#         keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
#         keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
#     ],
# )
# _, accuracy, top_5_accuracy = model.evaluate(test, test_label)
# print(f"Test accuracy: {round(accuracy * 100, 2)}%")
# print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")









