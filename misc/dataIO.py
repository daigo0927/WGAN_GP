# coding:utf-8

import numpy as np
from tqdm import tqdm
from glob import glob

from .utils import *

class InputSampler:

    def __init__(self, datadir,
                 target_size = 108, image_size = 64,
                 split = 5, num_utilize = np.inf):
        
        self.datadir = datadir
        self.target_size = target_size
        self.image_size = image_size
        self.split = split

        self.image_paths = []
        for d in self.datadir:
            self.image_paths += glob(d + '/*')
        self.data_size = min(len(self.image_paths), num_utilize)
        print('data size : {}'.format(self.data_size))
        self.image_paths = np.random.choice(self.image_paths,
                                            self.data_size,
                                            replace = False)
        self.data = None

    def load(self):
        self.reload()

    def reload(self):
        image_paths_split = np.random.choice(self.image_paths,
                                             int(self.data_size/self.split),
                                             replace = False)
        print('split data loading ...')
        self.data = np.array([get_image(p, self.target_size, self.image_size)
                              for p in tqdm(image_paths_split)])

    def image_sample(self, batch_size):
        images = self.data[np.random.choice(len(self.data),
                                            batch_size,
                                            replace = False)]
        return images

    def noise_sample(self, batch_size, noise_dim = 100):
        noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
        return noise
        
        
