# coding:utf-8

import os, sys
import numpy as np
import pdb
from PIL import Image
import h5py
import argparse
from tqdm import tqdm

import keras.backend as K
import tensorflow as tf

from model import GeneratorDeconv, Discriminator
from misc.utils import combine_images
from misc.dataIO import InputSampler

def main():
    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help = 'number of epochs [20]')
    parser.add_argument('--lr_g', type = float, default = 1e-4,
                        help = 'learning rate for generator [1e-4]')
    parser.add_argument('--lr_d', type = float, default = 1e-4,
                        help = 'learning rate for discriminator [1e-4]')
    parser.add_argument('--train_size', type = int, default = np.inf,
                        help = 'size of trainind data [np.inf]')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    parser.add_argument('--nd', type = int, default = 5,
                        help = 'training schedule for dicriminator by generator [5]')
    parser.add_argument('--generator', type = str, default = 'deconv',
                        choices = ['deconv'],
                        help = 'choose generator type [deconv]')
    # data {/O
    parser.add_argument('--target_size', type = int, default = 108,
                        help = 'target area of training data [108]')
    parser.add_argument('--image_size', type = int, default = 64,
                        help = 'size of generated image [64]')
    parser.add_argument('--datadir', type = str, nargs = '+', required = True,
                        help = 'path to directory contains training (image) data')
    parser.add_argument('--split', type = int, default = 5,
                        help = 'load data, by [5] split')
    parser.add_argument('--loadweight', type = str, default = False,
                        help = 'path to directory conrtains trained weights [False]')
    parser.add_argument('--modeldir', type = str, default = './model',
                        help = 'path to directory put trained weighted [self./model]')
    parser.add_argument('--sampledir', type = str, default = './image',
                        help = 'path to directory put generated image samples [./image]')
    args = parser.parse_args()

    print('epochs : {}, lr_g : {}, lr_d : {}\n'.format(args.epochs, args.lr_g, args.lr_d),
          'train size : {}, batch size : {}, disc-schedule : {}\n'\
          .format(args.train_size, args.batch_size, args.nd),
          'generator type : {},'.format(args.generator),
          'target size : {}, image size : {}\n'.format(args.target_size, args.image_size),
          'data dir : {}\n,'.format(args.datadir),
          'load data splitingly : {}\n'.format(args.split),
          'weight flag : {}, model dir : {}, sample dir : {}'\
          .format(args.loadweight, args.modeldir, args.sampledir))

    disc = Discriminator(args.image_size)
    gen = GeneratorDeconv(args.image_size)

    wgan = WassersteinGAN(gen = gen, disc = disc,
                          z_dim = 100, image_size = args.image_size,
                          lr_d =  args.lr_d,
                          lr_g =  args.lr_g)

    sampler = InputSampler(datadir = args.datadir,
                           target_size = args.target_size, image_size = args.image_size,
                           split = args.split, num_utilize = args.train_size)

    wgan.train(nd = args.nd,
               sampler = sampler,
               epochs = args.epochs,
               batch_size = args.batch_size,
               sampledir = args.sampledir,
               modeldir = args.modeldir)
    

class WassersteinGAN:

    def __init__(self,
                 gen, disc,
                 z_dim, image_size,
                 lr_d, lr_g):

        self.gen = gen
        self.disc = disc
        self.z_dim = z_dim
        self.image_size = image_size

        self.x = tf.placeholder(tf.float32,
                                (None, self.image_size, self.image_size, 3),
                                name = 'x')
        self.z = tf.placeholder(tf.float32,
                                (None, self.z_dim),
                                name = 'z')
        self.x_ = self.gen(self.z)

        self.d = self.disc(self.x)
        self.d_ = self.disc(self.x_)

        self.d_loss = -(tf.reduce_mean(self.d) - tf.reduce_mean(self.d_))
        self.g_loss = -tf.reduce_mean(self.d_)

        # gradient penalty
        self.alpha = tf.random_uniform((tf.shape(self.x)[0], 1),
                                       minval = 0., maxval = 1,)
        self.differ = self.x_ - self.x
        self.interp = self.x + (self.alpha * self.differ)
        self.grads = tf.gradients(self.disc(self.interp), [self.interp])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.grads),
                                            reduction_indices = [3]))
        self.grad_penalty = tf.reduce_mean((self.slopes - 1.)**2)
        self.d_loss += 10 * self.grad_penalty

        self.lr_d = lr_d
        self.lr_g = lr_g

        self.d_opt = tf.train.AdamOptimizer(learning_rate = self.lr_d,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.d_loss, var_list = self.disc.trainable_weights)
        self.g_opt = tf.train.AdamOptimizer(learning_rate = self.lr_g,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.g_loss, var_list = self.gen.trainable_weights)

        self.sess = tf.Session()
        K.set_session(self.sess)

    def train(self,
              nd, sampler,
              batch_size, epochs,
              sampledir, modeldir):
        
        num_batches = int(sampler.data_size/batch_size)
        print('epochs : {}, number of batches : {}'.format(epochs, num_batches))

        self.sess.run(tf.global_variables_initializer())

        # training iteration
        for e in range(epochs):
                
            for batch in range(num_batches):

                if batch in np.linspace(0, num_batches, sampler.split+1, dtype = int):
                    sampler.reload()

                d_iter = nd
                if batch%500 == 0 or batch<25:
                    d_iter = 100

                for _ in range(d_iter):
                    d_weights = [np.clip(w, -0.01, 0.01) for w in self.disc.get_weights()]
                    self.disc.set_weights(d_weights)

                    bx = sampler.image_sample(batch_size)
                    bz = sampler.noise_sample(batch_size)
                    self.sess.run(self.d_opt, feed_dict = {self.x: bx, self.z: bz,
                                                           K.learning_phase(): 1})

                bz = sampler.noise_sample(batch_size, self.z_dim)
                self.sess.run(self.g_opt, feed_dict = {self.z: bz,
                                                       K.learning_phase(): 1})

                if batch%10 == 0:
                    d_loss, g_loss = self.sess.run([self.d_loss, self.g_loss],
                                                   feed_dict = {self.x: bx, self.z: bz,
                                                                K.learning_phase(): 1})
                    print('epoch : {}, batch : {}, d_loss : {}, g_loss : {}'\
                          .format(e, batch, d_loss, g_loss))

                if batch%100 == 0:
                    fake_sample = self.sess.run(self.x_, feed_dict = {self.z: bz,
                                                                      K.learning_phase(): 1})
                    fake_sample = combine_images(fake_sample)
                    fake_sample = fake_sample*127.5 + 127.5
                    Image.fromarray(fake_sample.astype(np.uint8))\
                         .save(sampledir + '/sample_{}_{}.png'.format(e, batch))

            self.gen.save_weights(modeldir + '/g_{}epoch.h5'.format(e))
            self.disc.save_weights(modeldir + '/d_{}epoch.h5'.format(e))

if __name__ == '__main__':
    main()
