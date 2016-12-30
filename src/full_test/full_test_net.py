# A. Lons
# December 2016
#
# Place where all the general network compontnets are placed.


import tensorflow as tf






# I want to add variable summaries as well here!
def variable_summaries(var):
    """
    DESCRIPTION
    Takes in the variables and adds a summary to it for tensorboard visulaizations,
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
        tf.summary.histogram('histogram', var)


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


