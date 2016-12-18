# A. Lons
# December 2016
#
# Just the code from https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
# Which is Imanol Schlag's blog. TensorFlow documents on this are fucking confusing so I have to look else where.
#
# TensorFlow Input Pipelines for Large Data Sets
# ischlag.github.io
# TensorFlow 0.11, 07.11.2016
#
# "This is a toy example ..., TensorFlow makes it easy to read data from TFRecods files. If you can convert your data
# into this binary format you might want to. In this example we are not doing that. In this example we are going to use
# two queues. One will be fed image-paths and labels by our own enqueue functions reading from a numpy array (or file
# if you wanted). The second one is hidden inside tf.train.batch which will dequeue samples from our first queue, load
# the image, and do some processing and build the batches.


import tensorflow as tf
import numpy as np
import threading

# Generating some simple data,
# This numpy data will fit in RAM, but instead of readinf from a numpy array you could change it to rad directly from a
# if you can't fit the whole dataset in memory.
r = np.arange(0.0, 100003.0)
raw_data = np.dstack((r, r, r, r))[0]                   #100003 X 4 numpy.array
raw_target = np.array([[1, 0, 0]] * 100003)             #100003 X 3 numpy.array


# are used to feed data into our queue
queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])

# From our numpy data, we will read multiple samples as once and push them into our FIFO queue. For this purpose
# we have the placeholders above, the queue itself, aldong with the enqueue and dequeue operations.
# From TFs api,
#   capacity = An integr, the upper bound on the number of elements stored
#   ...
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]])   # The queue
enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])                 # queue operation
dequeue_op = queue.dequeue()                                                            # queue operation


# After performing some preprocessing on the dequeueed data, we can group them into a bathc and use a session in order
# to draw the next batch of samples from our input pipeline. But before we can do that, we have to start a thred that
# will fill our queue object by calling queue.enqueue_many with data from our numpy data. Here instead of reading
# from our simple numpy data you could do this from a big file. Notice I loop endlessly in order to keep up a
# stream of incoming data.
#
# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
# use this to shuffle batches:
# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)


# Now the only thing missing are the queue runner threads fro our tf.train.batch, that is what this method will do.
#...... (AJL) I think this replace essentially a tf.train.QueueRunner() object
def enqueue(sess):
  """ Iterates over our data puts small junks into our queue."""
  under = 0
  max   = len(raw_data)
  while True:
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
      curr_data = raw_data[under:upper]
      curr_target = raw_target[under:upper]
      under = upper
      print(type(curr_data))
    else:
      rest = upper - max
      curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
      curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
      under = rest
      print(type(curr_data))


    sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                    queue_input_target: curr_target})
    print("added to the queue")
  print("finished enqueueing")

# start the threads for our FIFOQueue and batch
sess = tf.Session()
enqueue_thread = threading.Thread(target=enqueue, args=[sess])
enqueue_thread.isDaemon()
enqueue_thread.start()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Fetch the data from the pipeline and put it where it belongs (into your model)
for i in range(5):
  run_options = tf.RunOptions(timeout_in_ms=4000)
  curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
  #print(curr_data_batch)

# shutdown everything to avoid zombies
sess.run(queue.close(cancel_pending_enqueues=True))
coord.request_stop()
coord.join(threads)
sess.close()