# Alex Lonsberry
# November 2016
#
# Within is code associated with the "Basic Usage" online for Tensor Flow documentation
# see https://www.tensorflow.org/versions/r0.11/get_started/basic_usage.html#overview

# "Building the graph"

import tensorflow as tf

########################################################################################################################
# Building the graph & Launching the graph in a session

# Create a Constant op that produces a 1X2 matrix. The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create anothor Constatn that produces a 2X1 matrix
matrix2 = tf.constant([[2.],[2.]])

# Creat a Matmul op that takes 'matrix1' and 'matrix2' an inputs. The returned value, 'product' represents the
# results of the matrix multiplication
product = tf.matmul(matrix1,matrix2)

# Our grpah now has three nodes, to actually run this we must launch the graph in a 'session'

# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
results = sess.run(product)
print(results)

# Close the Session when we're done.
sess.close()

# An alternative way to run a session is with a 'with'black where the session
# will close automatically
if __name__ == '__main__':
    if __name__ == '__main__':
        with tf.Session() as sess:
            result = sess.run(product)
            print(result)

########################################################################################################################
# Variables

# Variable maintain state across executions of the graph

# Create a variable that will be initialized to a scalar value 0
state = tf.Variable(0., name="counter")

# Create an Op to add one to 'state'

one = tf.constant(1.)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#Varaiables must be init. by running an 'init' Op after having launched the graph.
init_op = tf.initialize_all_variables()

#Launch the graph and run the ops
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    # Prin the initial value
    print('\n')
    print(sess.run(state))
    # Run
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

########################################################################################################################
# Fetches

# To Fetch output of operations, execute the graph with run() call on the Session object an pass in the tensors to
# retrieve.

input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)
print("\n")
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)


########################################################################################################################
# Feeds

# The feed mechanism provides a mechanism for patching a tensor directly into any operation in the graph.

# A feed temporarily replaces the output of an operation with a tensor value. You supply feed data as an argument to a
# run() call. The feed is only used for the run call to which it is passed. The most common use cases involves
# designating specific operations to be "feed" operations

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
print('\n')
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.0], input2:[2.0]}))
