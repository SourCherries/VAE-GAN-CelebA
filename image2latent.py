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

file = "example.jpg"

sess, X, G, Z, Z_mu, is_training, saver = make_network.make_network()
if os.path.exists("vaegan_celeba.ckpt"):
    saver.restore(sess, "vaegan_celeba.ckpt")
    print("VAE-GAN model restored.")
else:
    print("Pre-trained network appears to be missing.")
    sys.exit()

img = plt.imread(file)[..., :3]
img = utils.preprocess128(img,crop_factor=0.8)[np.newaxis]
                
#generate images from z
z = sess.run(Z_mu, feed_dict={X: img, is_training: False})

#save data in Matlab format
savemat(file[:-4]+'_z',dict(latent=z))
