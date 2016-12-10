# Alex Lonsberry
# December 2016
#
# DESCRIPTION
# From sentdex youutube playlist/vidoes "Machine Learning with Python"
# The video title is "TensorFLow Basics - Deep learning with Neural Networks p. 2"
#
# This is the first video that comes up after the first video on installation.

import tensorflow as tf

# First thing we do is construct the graph, make some constatns
x1 = tf.constant(5)
x2 = tf.constant(6)
# Probably always most efficient to call and use built in tf functions
result = tf.mul(x1, x2)
# What is result now.... its not what you expect
print()
print(result)
# The result is a tensor, nothing has been run, essentially you have just set up an operation (op). Thus not just like
# you think it would be if this was regular python.

# 'Session' is the way to actually run graph
sess = tf.Session() #Gives a session object
print()
print(sess.run(result))
# WE ACTUALLY GOT THE GRAPH TO RUN
sess.close()
# What we see is that we need to run a session for any operations in the graph to actually run. We need to ALWAYS
# run the .close() operation.

# Alternative way, with automatic closing
with tf.Session() as sess:
    print()
    output = sess.run(result)
    print(output)
# Can we access output now? Somehow we are (I am not sure how output lives outside the scope...)
print(output)


