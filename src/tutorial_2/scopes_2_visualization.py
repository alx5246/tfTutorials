# A. Lons
# December 2016
#
# DESCRIPTION
# I am going to work on visulization of a graph, while it does not seem like it should be part of my scope tutorials,
# the graph visualization is completely based on nameing scopes appropriatesly
#
# For this I am going to start by reusing the code I generated in scopes_1.py which is a ANN for the mnist
#
# This is built for tensorflow 0.12.rc1


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # This is where we will get our example data.

# Load in the data (somehow using built in tensor flow)
# "one_hot" means 1 is on and the rest are off (in circiuts), we have 10 classes, and calling this "one_hot" is to
# be such that we have 10 outputs (one for each class). We need to make sure we can load all this in. For other
# dataset we need to be smart about how to load in based on size of ram.
mnist = input_data.read_data_sets("/home/alex/pythonCode/tensorFlowTutorials/src/tutorial_1/", one_hot=True)

# Define number of nodes for each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of classes
n_classes = 10

# Batch size, this is how many inputs we give at once.
batch_size = 100


# I want to give this input a name so it will show up on my graph!
with tf.name_scope('input'):
    # Define some placeholders, x will be input (flatten). Notes I think the tf.placeholder still constitutes an 'op'
    x = tf.placeholder('float', [None, 784], name='x-input') # Flatten input images.
    y = tf.placeholder('float', name='y-input') # This will be the label


# I want to add variable summaries as well here!
def variable_summaries(var):
    """
    DESCRIPTION
    Takes in the variables and adds a summary to it for tensorboard visulaizations
    :param var:
    :return:
    """
    # Remember name_scopes inheret
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram',var)
        #tf.summary.histogarm('histogram', var)


def gen_hidden_layer(input, kernel_shape, bias_shape, layer_name):
    #
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # Generate weights
            weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
            variable_summaries(weights)
        with tf.name_scope("biases"):
            # Generate bias terms
            biases = tf.get_variable("biases", bias_shape, initializer=tf.random_normal_initializer())
            variable_summaries(biases)
        with tf.name_scope('preActivation'):
            preactive = tf.add(tf.matmul(input, weights), biases)
            #tf.summary.histogarm('preActivations', preactive)
        # Calculate output
        output = tf.nn.relu(preactive)
        #tf.hsummary.histogarm('activations', output)
        # Send out
    return output


def gen_output_layer(input, kernel_shape, bias_shape, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # Generate weights
            weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
        with tf.name_scope("biases"):
            # Generate bias terms
            biases = tf.get_variable("biases", bias_shape, initializer=tf.random_normal_initializer())
        with tf.name_scope("output"):
            # Calculate output
            output = tf.add(tf.matmul(input, weights), biases)
        # Send out
    return output


# Now define neural network model, this is essentially the computation graph (tha majority of it)
def neural_network_model(data):
    # Now create our hidden layers!
    with tf.variable_scope("hidden_1"):
        h1 = gen_hidden_layer(data, [784, n_nodes_hl1], [n_nodes_hl1], "hidden_1")
    with tf.variable_scope("hidden_2"):
        h2 = gen_hidden_layer(h1, [n_nodes_hl1, n_nodes_hl2], [n_nodes_hl2], "hidden_2")
    with tf.variable_scope("hidden_3"):
        h3 = gen_hidden_layer(h2, [n_nodes_hl2, n_nodes_hl3], [n_nodes_hl3], "hidden_3")
    with tf.variable_scope("output"):
        output = gen_output_layer(h3, [n_nodes_hl3, n_classes], [n_classes], "output")
    return output


# Now define loss
def loss_func(prediction, y):
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(prediction, y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        #tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


# Now define optimizer
def optim_func(cross_entropy):
    with tf.name_scope("train"):
        trian_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    return trian_step


# Now define accuracy calc
def accuracy_calc(prediction, y):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #tf.summary.scalar('accuracy', accuracy)
    return accuracy


# Now generate the entire model!
prediction = neural_network_model(x)
cost = loss_func(prediction, y)
optimizer = optim_func(cost)
accuracy = accuracy_calc(prediction=prediction, y=y)





# Now define the feed dictionary
def feeder(train):
    if train:
        epoch_x, epoch_y = mnist.train.next_batch(100)
    else:
        epoch_x, epoch_y = mnist.test.images, mnist.test.labels
    return {x: epoch_x, y: epoch_y}


# Now we have modeled the neural network! This means we are basically done with the computational graph. We now have to
# define some other stuff and what to do with the model.
Runs = 100
with tf.Session() as sess:

    #Still not sure when to use gloabl and when not too
    tf.global_variables_initializer().run()

    # Now prepare all summaries
    #merged = tf.merge_all_summaries()
    merged = tf.summary.merge_all()
    train_write = tf.summary.FileWriter('traingraph',sess.graph)
    test_write = tf.summary.FileWriter('testgraph')


    for i in range(Runs):
        if i % 10 == 0: # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feeder(False))
            test_write.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            summary, _ = sess.run( [merged, optimizer], feed_dict=feeder(True) )
            train_write.add_summary(summary, i)


train_write.close()
test_write.close()