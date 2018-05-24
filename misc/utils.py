# coding:utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.misc import imread, imresize



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


def show_progress(epoch, batch, batch_total, loss, accuracy):
    sys.stdout.write(f'\r{epoch} epoch: [{batch}/{batch_total}, loss: {loss}, acc: {accuracy}]')
    sys.stdout.flush()
