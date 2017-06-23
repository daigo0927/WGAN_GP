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
def GeneratorDeconv(image_size = 64): 

    L = int(image_size)

    inputs = Input(shape = (100, ))
    x = Dense(512*int(L/16)**2)(inputs) #shape(512*(L/16)**2,)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/16), int(L/16), 512))(x) # shape(L/16, L/16, 512)
    x = Conv2DTranspose(256, (4, 4), strides = (2, 2),
                        padding = 'same')(x) # shape(L/8, L/8, 256)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (4, 4), strides = (2, 2),
                        padding = 'same')(x) # shape(L/4, L/4, 128)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides = (2, 2),
                        padding = 'same')(x) # shape(L/2, L/2, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(3, (4, 4), strides= (2, 2),
                        padding = 'same')(x) # shape(L, L, 3)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)
    model.summary()
    return model


def Discriminator(image_size = 64):

    L = int(image_size)

    images = Input(shape = (L, L, 3))
    x = Conv2D(64, (4, 4), strides = (2, 2), padding = 'same')(images) # shape(L/2, L/2, 32)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides = (2, 2), padding = 'same')(x) # shape(L/4, L/4, 64)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides = (2, 2), padding = 'same')(x) # shape(L/8, L/8, 128)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4, 4), strides = (2, 2), padding = 'same')(x) # shape(L/16, L/16, 256)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs = images, outputs = outputs)
    model.summary()
    return model
