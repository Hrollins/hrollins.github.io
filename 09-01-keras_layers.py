# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:49:04 2022

@author: HRollins
"""
import tensorflow as tf
from tensorflow.keras import layers

#%% Dense Layer Example
input_shape = (10,20)
x = tf.random.uniform(input_shape) #We have 10 sample and each has 20 features

denseLayer1 = layers.Dense(32, activation='relu')
denseLayer2 = layers.Dense(25, activation='relu')

y1 = denseLayer1(x)
y2 = denseLayer2(y1)

print("input:      ",x.shape)
print("dense1 out: ",y1.shape)
print("dense2 out: ",y2.shape)

#%% Conv Layer Example

input_shape = (4, 28, 28, 3) #we have 4 samples and each has 28x28x3 size
x = tf.random.uniform(input_shape)

convLayer1 = layers.Conv2D(2, kernel_size=(3, 3), activation='relu') #we create 2 3x3 kernels
convLayer2 = layers.Conv2D(5, kernel_size=(7, 7), activation='relu') #we create 5 7x7 kernels

y1 = convLayer1(x)
y2 = convLayer2(y1)

print("input:     ",x.shape)
print("conv1 out: ",y1.shape)
print("conv2 out: ",y2.shape)

#%% Maxpool Layer Example

x = tf.constant([[1., 2., 3., 4.], 
                 [5., 6., 7., 8.], 
                 [9., 10., 11., 12.],
                 [13., 14., 15., 16.]])

x = tf.reshape(x, [1, 4, 4, 1]) #we have 1 samples and each has 3x4x1 size

mpLayer1 = layers.MaxPooling2D(pool_size=(2, 2)) #we create 2x2 maxpool layer
mpLayer2 = layers.MaxPooling2D(pool_size=(2, 2))

y1 = mpLayer1(x)
y2 = mpLayer2(y1)

print("input:     ",x.shape)
print("maxPool1 out: ",y1.shape)
print("maxPool2 out: ",y2.shape)





