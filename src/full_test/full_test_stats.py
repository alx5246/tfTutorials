#
#
#

import tensorflow as tf

def accuracy_calc(prediction, actual):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_predition'):
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual, 1), name='find_correct')
        with tf.name_scope('acc_vals'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='find_avg_correct')
    return accuracy
