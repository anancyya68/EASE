# @File : networks.py
# @Time : 2024/5/20 17:24
# @Author :
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
#from utils import *
# from distribution_shift import DistributionShiftEvaluator
# from feature_selection import FeatureSelector
import random
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer, Dense, Input,Reshape,Flatten,Concatenate,LSTM, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import *
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Attention
class PreAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super(PreAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_dense = Dense(embed_dim)
        self.k_dense = Dense(embed_dim)
        self.v_dense = Dense(embed_dim)
        self.out_dense = Dense(embed_dim)

    def build(self, input_shape):
        super(PreAttention, self).build(input_shape)
        self.q_dense.build(input_shape)
        self.k_dense.build(input_shape)
        self.v_dense.build(input_shape)
        self.out_dense.build(input_shape)


    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_weights = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = attention_weights / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        output = self.out_dense(concat_output)
        return output,attention_weights


# NN
class NeuralNetwork:
    def __init__(self, num_class):
        self.num_class = num_class

    def mlp_method(self, input_data,embed_dim):
        dense1 = Dense(512, activation='relu')
        dense2 = Dense(256, activation='relu')
        dense3 = Dense(128, activation='relu')
        dense_output = Dense(self.num_class*embed_dim, activation='softmax')
        x = dense1(input_data)
        x = dense2(x)
        x = dense3(x)
        output = dense_output(x)
        return  output

    def lstm_method(self):
        pass

    def cnn_method(self):
        pass


# combine model
def combined_model(embed_dim, num_class, num_head):
    inputs = Input(shape=(None, embed_dim))
    # 1.attention
    att = PreAttention(embed_dim, num_head)
    attention_output, attention_weights = att.call(inputs)

    # 2.concat input
    combined_output = Concatenate(axis=-1)([inputs, attention_output])
    # # 3.different shape
    # lstm_output = LSTM(256, return_sequences=True)(combined_output)
    global_avg_pooling = GlobalAveragePooling1D()(combined_output)
    # 4.NN layer
    neural_net = NeuralNetwork(num_class)

    nn_output = neural_net.mlp_method(global_avg_pooling,embed_dim)
    output = Reshape((embed_dim, num_class))(nn_output)
    model = Model(inputs=inputs, outputs=output)
    return model
