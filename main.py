# coding:utf-8

import os, sys
import numpy as np
import pdb
from PIL import Image
import h5py
import argparse

import keras.backend as K
import tensorflow as tf

from model import GeneratorDeconv, Discriminator
from misc.utils import *
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
    parser.add_argument('-d', '--datadir', type = str, nargs = '+', required = True,
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

    for key, item in vars(args).items():
        print(f'{key} : {args}')

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

        self.d = tf.reduce_mean(self.disc(self.x))
        self.d_ = tf.reduce_mean(self.disc(self.x_))

        self.d_loss = -(self.d - self.d_)
        self.g_loss = -self.d_

        # gradient penalty
        alpha = tf.random_uniform((tf.shape(self.x)[0], 1, 1, 1),
                                  minval = 0., maxval = 1,)
        differ = self.x_ - self.x
        interp = self.x + (alpha * differ)
        grads = tf.gradients(self.disc(interp), [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                       reduction_indices = [3]))
        grad_penalty = tf.reduce_mean((slopes - 1.)**2)
        self.d_loss += 10 * grad_penalty

        self.lr_d = lr_d
        self.lr_g = lr_g

        self.d_opt = tf.train.AdamOptimizer(learning_rate = self.lr_d,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.d_loss, var_list = self.disc.trainable_weights)
        self.g_opt = tf.train.AdamOptimizer(learning_rate = self.lr_g,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.g_loss, var_list = self.gen.trainable_weights)

        self.saver = tf.train.Saver()

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

                for _ in range(d_iter):
                    bx = sampler.image_sample(batch_size)
                    bz = sampler.noise_sample(batch_size)
                    self.sess.run(self.d_opt, feed_dict = {self.x: bx, self.z: bz,
                                                           K.learning_phase(): 1})

                bz = sampler.noise_sample(batch_size, self.z_dim)
                self.sess.run(self.g_opt, feed_dict = {self.z: bz,
                                                       K.learning_phase(): 1})

                if batch%10 == 0:
                    d_real, d_fake = self.sess.run([self.d, self.d_],
                                                   feed_dict = {self.x: bx, self.z: bz,
                                                                K.learning_rate(): 1})
                    show_progress(e+1, batch+1, num_batches, d_real-d_fake, None)


                if batch%100 == 0:
                    fake_seed = sampler.noise_sample(9, self.z_dim)
                    fake_sample = self.sess.run(self.x_, feed_dict = {self.z: fake_seed,
                                                                      K.learning_phase(): 1})
                    fake_sample = combine_images(fake_sample)
                    fake_sample = fake_sample*127.5 + 127.5
                    Image.fromarray(fake_sample.astype(np.uint8))\
                         .save(sampledir + '/sample_{}_{}.png'.format(e, batch))

            self.saver.save(self.sess, modeldir + '/result{}.ckpt'.format(e))


if __name__ == '__main__':
    main()
