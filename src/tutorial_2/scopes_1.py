# A. Lons
# December 2016
#
# DESCRIPTION
# Here I am going to try to use variable-scopes in an instantiation of an ANN. In particular I am going to work with
# the network and code in tutorial_1/tut_46_DeepWithNNwithTP.py which creates a simple ANN to handle mnist. I am going
# to modify the code so I can try out the variable_scopes.
#
# This is here to work on variable_scopes mostly and a little bit name_scopes.

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

# Define some placeholders, x will be input (flatten). Notes I think the tf.placeholder still constitutes an 'op'
x = tf.placeholder('float', [None, 784]) # Flatten input images.
y = tf.placeholder('float') # This will be the label


def gen_hidden_layer(input, kernel_shape, bias_shape):
    # Generate weights
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    print(weights.name)
    # Generate bias terms
    biases = tf.get_variable("biases", bias_shape, initializer=tf.random_normal_initializer())
    # Calculate output
    output = tf.nn.relu(tf.add(tf.matmul(input, weights), biases))
    # Send out
    return output


def gen_output_layer(input, kernel_shape, bias_shape):
    # Generate weights
    # Generate weights
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Generate bias terms
    biases = tf.get_variable("biases", bias_shape, initializer=tf.random_normal_initializer())
    # Calculate output
    output = tf.add(tf.matmul(input, weights), biases)
    # Send out
    return output


# Now define neural network model, this is essentially the computation graph (tha majority of it)
def neural_network_model(data):

    # Now create our hidden layers!
    with tf.variable_scope("hidden_1"):
        h1 = gen_hidden_layer(data, [784, n_nodes_hl1], [n_nodes_hl1])
    with tf.variable_scope("hidden_2"):
        h2 = gen_hidden_layer(h1, [n_nodes_hl1, n_nodes_hl2], [n_nodes_hl2])
    with tf.variable_scope("hidden_3"):
        h3 = gen_hidden_layer(h2, [n_nodes_hl2, n_nodes_hl3], [n_nodes_hl3])
    with tf.variable_scope("output"):
        output = gen_output_layer(h3, [n_nodes_hl3, n_classes], [n_classes])

    # # NOTE, we can call variables from outside scope of the method! This is somthing I do not think of regularily!
    #
    # # Make some dictionaries for our weights across the the layers. We define by numb inputs and numb outputs. In the
    # # case of bias, we only care about the number of hidden nodes. At this point I am not sure if I need to make
    # # dictionaries for weights... etc.
    # hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    # hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    # output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    #                 'biases': tf.Variable(tf.random_normal([n_classes]))}
    # # Now handle how we find the layer inputs and outputs, thus here we are defining tf 'ops' here.
    # # Sum inputs and then nonlinear activation funcion
    # l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # l1 = tf.nn.relu(l1)
    # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    # output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

# Now we have modeled the neural network! This means we are basically done with the computational graph. We now have to
# define some other stuff and what to do with the model.

# A method to train
def train_neural_network(x, y):

    # We are saying now that the prediction is running the former method above, that is we use our former method to
    # define the model
    prediction = neural_network_model(x)

    # Now we generate a cost function (so tf knows what this is)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # No we have make an optimizer (adam default learning_rate = 0.001), NOTE there are other training methods
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Setup epochs
    hm_epochs = 10

    # Run tf 'session'
    with tf.Session() as sess:

        # We have to initialize the variables (all those that we have formally created, all created are part of the
        # basic graph)
        sess.run(tf.initialize_all_variables())

        # Run over the epochs
        for epoch in range(hm_epochs):

            epoch_loss = 0.

            # We are going to use a specified batch size.
            for _ in range(int(mnist.train.num_examples/batch_size)):

                # This magic command below automaticllay goes through data of specific batch size.
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # use session run function! to call optimizer
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x, y)