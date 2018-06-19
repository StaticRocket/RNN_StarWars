# Code to generate our final script for Star Wars Beyond after training

import tensorflow as tf
import numpy as np
import my_txtutils

ALPHA_SIZE = my_txtutils.ALPHASIZE
NUM_LAYERS = 3
NUM_OF_GRUS = 512

# Import weights
FinalCheckpoint = "checkpoints/StarWars_train_XXXXXXXXXX"
author = FinalCheckpoint
ncnt = 0
with tf.Session() as sess:
    # Import graph
    new_saver = tf.train.import_meta_graph('checkpoints/StarWars_train_1529351989-0.meta')
    new_saver.restore(sess, author)
    x = my_txtutils.convert_from_alphabet(ord("L"))
    x = np.array([[x]])

    y = x
    h = np.zeros([1, NUM_OF_GRUS * NUM_LAYERS], dtype=np.float32) 
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0
