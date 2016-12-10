# Alex Lonsberry
# December 2016
#
# DESCRIPTION
# From sentdex youutube playlist/vidoes "Training/Testing on our Data - Deep Learning with Neural Networks and Tensor
# Flow part 7", this is video #50 in his series.
#
# This part uses work we did in file tut_48_DeepWithNNwithTF.py (that is where we made our data for TF use)
#
#

import tensorflow as tf
import pickle
import numpy as np

# Load in the data from what was created before, (I am doing this differently than the instructor)
with open('sentiment_set.pickle', 'rb') as f:
    loadedData = pickle.load(f)

train_x = loadedData[0]
train_y = loadedData[1]
test_x = loadedData[2] # This seems to be a list of lists, ...
test_y = loadedData[3]
print(type(test_x))
print(len(test_x))
print(type(test_x[0]))
print(len(test_x[0]))
print(type(test_x[0][0]))


# Define number of nodes for each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of classes
n_classes = 2

# Batch size, this is how many inputs we give at once.
batch_size = 100

#
x = tf.placeholder('float', [None, len(train_x[0]) ]) # Flatten input images.
y = tf.placeholder('float') # This will be the label

# Now define neural network model, this is essentially the computation graph (tha majority of it)
def neural_network_model(data):

    # NOTE, we can call variables from outside scope of the method! This is somthing I do not think of regularily!

    # Make some dictionaries for our weights across the the layers. We define by numb inputs and numb outputs. In the
    # case of bias, we only care about the number of hidden nodes. At this point I am not sure if I need to make
    # dictionaries for weights... etc.
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Now handle how we find the layer inputs and outputs, thus here we are defining tf 'ops' here.

    # Sum inputs and then nonlinear activation funcion
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

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

            #
            epoch_loss = 0.

            # Make our own batches
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                # use session run function! to call optimizer
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                epoch_loss += c

                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x, y)