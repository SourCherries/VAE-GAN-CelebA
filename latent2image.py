#!/usr/bin/python3

# First check the Python version
import sys, getopt
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n',
          'You should consider updating to Python 3.4.0 or',
          'higher.\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import libs.make_network_deterministic as make_network
    import libs.utils as utils
    from scipy.io import loadmat, savemat
except ImportError as e:
    print("Make sure the libs folder is available in current directory.")
    print(e)

print('TF version = ',tf.__version__)

sys.path

file = "example_z"

sess, X, G, Z, Z_mu, is_training, saver = make_network.make_network()
if os.path.exists("vaegan_celeba.ckpt"):
    saver.restore(sess, "vaegan_celeba.ckpt")
    print("VAE-GAN model restored.")
else:
    print("Pre-trained network appears to be missing.")
    sys.exit()


#load data from Matlab format
latent = loadmat(file)['latent'] #make sure that the matlab variable name is 'latent'
print('input data to be transformed:',latent.shape)
if latent.shape[0]==1024:
    latent=latent[np.newaxis]
elif latent.shape[1]!=1024:
    print("None of the input dimensions appears to be 1024!!!")
                
#generate images from z
g = sess.run(G, feed_dict={Z: (1*latent), is_training: False})

def imdeprocess(g):
    stretch = 1.0
    for i in range(g.shape[0]):
        g[i]=np.clip(stretch*g[i] / (g.max()),0,1)
    return g

g=imdeprocess(g)
                
#if there's only one image, we save it as a jpg
if g.shape[0]==1:
    plt.imsave(file[:-4]+'_g.jpg',g[0])
                
#in all cases, we save the image(s) as a .mat file
savemat(file[:-4]+'_g',dict(images=g))


