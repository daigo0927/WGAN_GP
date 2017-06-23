# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import h5py

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D

# ConvTranspose (often called Deconv) ver.
def generator_deconv(image_size = 64): 

    L = int(image_size)

    inputs = Input(shape = (100, ))
    x = Dense(512*int(L/16)**2)(inputs) #shape(512*(L/16)**2,)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/16), int(L/16), 512))(x) # shape(L/16, L/16, 512)
    x = Conv2DTranspose(256, (2, 2), strides = (2, 2),
                        padding = 'same')(x) # shape(L/8, L/8, 256)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (2, 2), strides = (2, 2),
                        padding = 'same')(x) # shape(L/4, L/4, 128)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (2, 2), strides = (2, 2),
                        padding = 'same')(x) # shape(L/2, L/2, 64)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(3, (2, 2), strides= (2, 2),
                        padding = 'same')(x) # shape(L, L, 3)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)

    model.summary()

    return model

# ConvTranspose (often called Deconv) ver.
def generator_upsampling(image_size = 64): 

    L = int(image_size)

    inputs = Input(shape = (100, ))
    x = Dense(512*int(L/16)**2)(inputs) #shape(512*(L/16)**2,)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/16), int(L/16), 512))(x) # shape(L/16, L/16, 512)
    x = UpSampling2D((2, 2))(x) 
    x = Conv2D(256, (5, 5), padding = 'same')(x) # shape(L/8, L/8, 256)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(128, (5, 5), padding = 'same')(x) # shape(L/4, L/4, 128)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(64, (5, 5), padding = 'same')(x) # shape(L/2, L/2, 64)
    x = BatchNormalization(trainable = False)(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x) 
    x = Conv2D(3, (5, 5), padding = 'same')(x) # shape(L, L, 3)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)

    model.summary()

    return model

def discriminator(image_size = 64):

    L = int(image_size)

    images = Input(shape = (L, L, 3))
    x = Conv2D(32, (5, 5), strides = (2, 2), padding = 'same')(images) # shape(L/2, L/2, 32)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/4, L/4, 64)
    x = BatchNormalization(trainable = False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/8, L/8, 128)
    x = BatchNormalization(trainable = False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/16, L/16, 256)
    x = BatchNormalization(trainable = False)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (4, 4), padding = 'valid')(x)
    outputs = Flatten()(x)
    
    model = Model(inputs = images, outputs = outputs)

    model.summary()

    return model
