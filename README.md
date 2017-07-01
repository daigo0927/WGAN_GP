# Wasserstein GAN with gradient penalty
Keras model and tensorflow optimization of ["improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028)

please download image training data before learning.

```
python main.py --datadir (path to the directory contains images)
```

sample generated image (using [lsun dataset](https://github.com/fyu/lsun), church_outdoor category)

![generated image](https://github.com/daigo0927/WGAN_GP/blob/master/image/sample_9_900.png)

! it generate well-seems fake images, but the loss (both of genertor and of discriminator) getting larger along learning.

! so it may have some imcomplete points, want someone to check it, thanks.
