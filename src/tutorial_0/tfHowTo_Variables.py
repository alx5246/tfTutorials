# Alex Lonsberry
# November 29 2016
#
# I am gong through TF's 'How to' sections. The first of thses is "Variable: Creation, Initialization, Saving, and
# Loading"

import tensorflow as tf

########################################################################################################################
# CREATION

# When you train some model, "variables" are used to hold and update parameters. Variables are in-memory buffers
# containing tensors. The must be explicitly initialized.

# When you create a "Variable" you pass a "Tensor" as its initial value to the 'Variable()' constructor.
# Create two variables
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# Calling tf.Variable() adds seval ops to the graph:
#   A variable op that holds the variable value
#   An initialer op that sets the variable to its initial value
#   ....

########################################################################################################################
# Device Placement

# A 'variable' can be pinned to a particular device when it is created using a 'with tf.device(..)"' block

# Pin a variable to the GPU
#with tf.device("/cpu:0"):
#    v = tf .... unfinished line

########################################################################################################################
# INITIALIZATION

# Variable initializers must be run before other ops in your model can be run.

# Add an op to initialize the variables.
#init_op = tf.initialize_all_variables()

# Later when running the session we have to call the init_op
# sess = tf.Session()
# sess.run(init_op)

# You have to be careful when initializing, in particular if you want to initialize from another variable
# Here, initialize with a value from another variable,
w2 = tf.Variable(weights.initialized_value(), name='w2')
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_tiwice")
# We need to the initalizer after all the former
init_op = tf.initialize_all_variables()
# Lets see what happens here!
sess = tf.Session()
sess.run(init_op)

print(sess.run(w2))
print(sess.run(w_twice))

sess.close()


########################################################################################################################
# SAVE AND RESTORE

# The easiest way to save an restore a model is to use a tf.train.Saver object. The constructer add save and restore ops
# to the graph fro all, or a specified list, of the variables in the graph. The saver object provides methods to run
# these ops, specifying paths for the checkpoint files to write to or read from.

# Lets try this out

# Create some variables
v1 = tf.Variable([3.0], name="v1")
v2 = tf.Variable([2.0], name="v2")

# Add initializer object
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch, init, and save..
sess = tf.Session()
sess.run(init_op)
save_path = saver.save(sess, "/home/alex/pythonCode/tensorFlowTutorials/src/tutorial_0/model.ckpt")
print("\n\nModel saved in file: %s" % save_path)
print("\n")

# BE CAREFUL, SEE THE w2 FROM BEFORE IS STILL AROUND!
#print(sess.run(w2))
#print(sess.run(v1))

# Now we can restore the variables

# The Saver object is used to restore variables. Note, that when yu restore variables from a file you do not have to
# initialize them beforehand.

saver.restore(sess, "/home/alex/pythonCode/tensorFlowTutorials/src/tutorial_0/model.ckpt")
print("\nModel restored")
print(sess.run(v1))

# You can selectively save what you want.
