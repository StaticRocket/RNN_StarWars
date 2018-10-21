# Star Wars Beyond Script Generator Trainer
# Takes in the scripts for Star Wars Episodes
# 1-7, and trains on them.

# TensorFlow Imports
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn 

# Standard Python Imports
import numpy as np
import math
import os
import time
import datetime

# Gorner Text Utilities Import
import my_txtutils as txt

# Start Time
start_time = str(datetime.datetime.now().time())

########################################
# HYPERPARAMETERS: Tweak settings here #
########################################
ALPHA_SIZE = txt.ALPHASIZE # 98 Upper, lower, numbers, symbols, etc
SEQ_LEN = 64 # Number of characters per sequence

NUM_EPOCHS = 50  # Number of epochs
BATCH_SIZE = 250  # Sequences per batch
NUM_OF_GRUS = 1024  # Number of GRU cells per layer
NUM_LAYERS = 3  # How many layers deep we are going

SET_LR = 0.001  # Small fixed learning rate
SET_PKEEP = 0.75    # Dropping 20% of neurons

# Seed our random number generator
tf.set_random_seed(0)

# Load our Star Wars Scripts.
filedir = "StarWarsScripts/*.txt"
traintext, validtext, scriptranges = txt.read_data_files(filedir, validation=True)

# Print out information about our data
size_of_epoch = len(traintext) // (BATCH_SIZE * SEQ_LEN)
txt.print_data_stats(len(traintext), len(validtext), size_of_epoch)

# Create our TensorFlow Graph.
batchsize = tf.placeholder(tf.int32, name='batchsize')
lr = tf.placeholder(tf.float32, name='lr')
pkeep = tf.placeholder(tf.float32, name='pkeep')
X = tf.placeholder(tf.uint8, [None, None], name='X') # Input vector
Xo = tf.one_hot(X, ALPHA_SIZE, 1.0, 0.0) # One Hots create vector size ALPHA_SIZE, all set 0 except character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_') # Output tensor
Yo_ = tf.one_hot(Y_, ALPHA_SIZE, 1.0, 0.0) # OneHot our output  also
Hin = tf.placeholder(tf.float32, [None, NUM_OF_GRUS*NUM_LAYERS], name='Hin') # Recurrent input states
cells = [rnn.GRUCell(NUM_OF_GRUS) for _ in range(NUM_LAYERS)] # Create all our GRU cells per layer
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells] # DropOut inside RNN
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # DropOut for SoftMax layer
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin) # Unrolling through time happens here
H = tf.identity(H, name='H')  # Last state of sequence
Yflat = tf.reshape(Yr, [-1, NUM_OF_GRUS])
Ylogits = layers.linear(Yflat, ALPHA_SIZE)
Yflat_ = tf.reshape(Yo_, [-1, ALPHA_SIZE])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
loss = tf.reshape(loss, [batchsize, -1])
Yo = tf.nn.softmax(Ylogits, name='Yo')
Y = tf.argmax(Yo, 1)
Y = tf.reshape(Y, [batchsize, -1], name="Y")
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Calculate Statistics for Analysis
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# TensorBoard Initialization
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Store checkpoints, make sure to create directory if it doesn't exist
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCH_SIZE * SEQ_LEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# Initialize and get ready to run our TensorFlow graph
istate = np.zeros([BATCH_SIZE, NUM_OF_GRUS *NUM_LAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

graph_writer = tf.summary.FileWriter(("log/" + timestamp + "-graph"), sess.graph)

# MAIN TRAINING LOOP
for x, y_, epoch in txt.rnn_minibatch_sequencer(traintext, BATCH_SIZE, SEQ_LEN, nb_epochs=NUM_EPOCHS):

    # Feed in and train one batch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: SET_LR, pkeep: SET_PKEEP, batchsize: BATCH_SIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

    # Store training TensorBoard data
    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCH_SIZE}
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, scriptranges, bl, acc, size_of_epoch, step, epoch)
        summary_writer.add_summary(smm, step)

    # Validation is batched in order to run quicker
    if step % _50_BATCHES == 0 and len(validtext) > 0:
        VALI_SEQ_LEN = 1*1024
        bsize = len(validtext) // VALI_SEQ_LEN
        txt.print_validation_header(len(traintext), scriptranges)
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(validtext, bsize, VALI_SEQ_LEN, 1))
        vali_nullstate = np.zeros([bsize, NUM_OF_GRUS*NUM_LAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,
                     batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        validation_writer.add_summary(smm, step) # Store validation TensorBoard data

    # Create teaser display text, should get better the more batches we perform
    if step // 3 % _50_BATCHES == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, NUM_OF_GRUS * NUM_LAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    # Save checkpoints, will also be helpful when running final network
    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/StarWars_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

    # Update progress bar
    progress.step(reset=step % _50_BATCHES == 0)

    # Increment steps and set input to output to prep for next loop
    istate = ostate
    step += BATCH_SIZE * SEQ_LEN

# TIME ELAPSED
print("\n")
print("Start: " + start_time)
print("End: " + str(datetime.datetime.now().time()))
