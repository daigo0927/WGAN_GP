# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.misc import imread, imresize

from keras.datasets import cifar10

def cifar10_extract(label = 'cat'):
    # acceptable label
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    target_label = labels.index(label)

    (x_train, t_train), (x_test, t_test) = cifar10.load_data()

    t_target = t_train==target_label
    t_target = t_target.reshape(t_target.size)

    x_target = x_train[t_target]
    
    print('extract {} labeled images, shape(5000, 32, 32, 3)'.format(label))
    return x_target


# shape(generated_images) : (sample_num, w, h, 3)
def combine_images(generated_images):

    total, width, height, ch = generated_images.shape
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)

    combined_image = np.zeros((height*rows, width*cols, 3),
                              dtype = generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1), :]\
            = image

    return combined_image

def get_image(filepath, image_target, image_size):
    
    img = imread(filepath).astype(np.float)
    h_origin, w_origin, _ = img.shape
    h_drop = int((h_origin - image_target)/2)
    w_drop = int((w_origin - image_target)/2)
    img_crop = img[h_drop:h_drop+image_target, w_drop:w_drop+image_target, :]
    img_resize = imresize(img_crop, [image_size, image_size])

    return np.array(img_resize)/127.5 - 1.
