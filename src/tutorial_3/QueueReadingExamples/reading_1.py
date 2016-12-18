# A. Lons
# December 2016
#
# DESCRIPTION
# Woring on understandin queues, threads, and loading data into tensorflow in a distributed form. For this tutorial
# and example I am working with stuff found on stackoverflow.com. In particular see,
# http://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow
# as well as,
# https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
# This has b

import numpy as np
import tensorflow as tf
import threading
import pickle

BATCH_SIZE = 2
TRAINING_ITERS = 220

# Input into and out of the queue
feature_input = tf.placeholder(tf.float32, shape=[5])
label_input = tf.placeholder(tf.float32, shape=[3])

# The queue
q = tf.FIFOQueue(50, [tf.float32, tf.float32], shapes=[[5], [3]])
enqueue_op = q.enqueue([feature_input, label_input])
feature_batch, label_batch = q.dequeue_many(BATCH_SIZE)

#..... some tensorflow function here

# Get the list of files we want to import!
from os import listdir
listOfFiles = listdir("npDataFiles")


sess = tf.Session()

def load_and_enqueue(sess, enqueue_op, coordinator):
    index = 0
    numbFiles = len(listOfFiles)
    while not coordinator.should_stop():
        if index < numbFiles:
            index = index
        else:
            index = 0
        f = open("npDataFiles/"+listOfFiles[index], "rb")
        feature_arr = pickle.load(f)
        label_arr = pickle.load(f)
        f.close()
        sess.run(enqueue_op, feed_dict={feature_input: feature_arr, label_input: label_arr})
        index += 1


coordinator = tf.train.Coordinator()
t = threading.Thread(target=load_and_enqueue, args=(sess, enqueue_op, coordinator))
t.start()


#
# Notes on Queue Runner class, if only adds an op to clsoe the queue, you still have to call the create_threads method
# to create threads! Thus a Queue-Runner does not have this on init. For example,
#   qr = tf.train.QueueRunner(queue_obj, ...)
# where qr, has no threads yet. To create threads, you have to call
#   someThreads = qr.create_threads(sess, coord=coord, ....)
# where sess is a session object, and coord is a coordinator object. My interpretation here is that I do not need to
# explicitly use QueueRunner objecets and its threads, but rather give some other generated threads, like,
#   t = threading.Thread(.....)
#


for i in range(TRAINING_ITERS):
    curr_feature, curr_label = sess.run([feature_batch, label_batch])
    print("\n\n")
    print(curr_feature)
    print(curr_label)

coordinator.request_stop()
coordinator.join([t])