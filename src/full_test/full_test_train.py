# A. Lons
# December 2016
#
# DESCRIPTION
#   The methods and code to run a full.
#
# I am relying on the example "fully-connected-reader" (see address below) it has a good example of how to connect
# different parts with coordinators and queue runners. I am also relying on "cifar10-multip-gpyu-trian" (see address
# below) which also seems to have an exmaple with queue runners and coordinator. In the standard cifar10-train I cannot
# find these entities.
#
# fully-connected-reader example see,
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py


import tensorflow as tf
import full_test_read_data as ftrd  # Hold the input pipei
import full_test_net as ftn
import full_test_stats as fts


def run_training():
    """

    :return:
    """

    # In both "fully-connected-reader" and "cifar10-multi-gpu-train" they call tf.Graph.as_defalul(). In the case of
    # the former, they say "tell TF that the model will be built into the default Graph."... I am not really sure what
    # this means. In the latter, they also call teh cup device (probably because later the call the GPU), but we can
    # add this too I guess
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        filenames = ['cifar-10-batches-bin/data_batch_1.bin',
                     'cifar-10-batches-bin/data_batch_2.bin',
                     'cifar-10-batches-bin/data_batch_3.bin',
                     'cifar-10-batches-bin/data_batch_4.bin',
                     'cifar-10-batches-bin/data_batch_5.bin']

        batch_size = 50

        queuing_threads = 4

        # Create input pipeline, that reads files, pulls examples, decodes them, batches them, and returns them here.

        imagesFlat, labels, key, imagesGray, imagesOrig  = ftrd.input_pipline(filenames, batch_size, queuing_threads)

        # Now create/build the graph that computes
        # Now define neural network model, this is essentially the computation graph (tha majority of it)
        # Now create our hidden layers!
        n_nodes_hl1 = 500
        n_nodes_hl2 = 500
        n_nodes_hl3 = 500
        n_classes = 10

        with tf.variable_scope("hidden_1"):
            h1 = ftn.gen_hidden_layer(imagesFlat, [1024, n_nodes_hl1], [n_nodes_hl1], "hidden_1")
        with tf.variable_scope("hidden_2"):
            h2 = ftn.gen_hidden_layer(h1, [n_nodes_hl1, n_nodes_hl2], [n_nodes_hl2], "hidden_2")
        with tf.variable_scope("hidden_3"):
            h3 = ftn.gen_hidden_layer(h2, [n_nodes_hl2, n_nodes_hl3], [n_nodes_hl3], "hidden_3")
        with tf.variable_scope("output"):
            net_output = ftn.gen_output_layer(h3, [n_nodes_hl3, n_classes], [n_classes], "output")

        # Now define loss
        with tf.name_scope("cross_entropy_loss"):
            diff = tf.nn.softmax_cross_entropy_with_logits(net_output , labels, name="loss_func")
            cross_entropy = tf.reduce_mean(diff, name='cross_entropy')

        # Now define optimizer
        with tf.name_scope('optimizer'):
            train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy, name='adam_optim')

        # Now define accuracy
        with tf.name_scope('accuracy'):
            accuracy = fts.accuracy_calc(net_output, labels)


        # Now we want to handle all the summaries and we have created!



        #
        # This is done the "fully_connected_reader" example. I seem to have to add the tf.local_variables_init()
        # because I set the num_epoch in the string producer in the other python file. I cannot just use the global
        # version it seems.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name="init_localglobal_vars")

        #
        sess = tf.Session()

        #
        sess.run(init_op)


        # Now prepare all summaries
        mergedSummaries = tf.summary.merge_all()
        trian_writer = tf.summary.FileWriter('trainingsum/train_summary', sess.graph)

        # Make coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Now use a 'try' in order to handle out of range errors! Simple
        try:
            step = 0
            while not coord.should_stop():
                # Run
                _, acc_value, summaryOut = sess.run([train_step, accuracy, mergedSummaries])
                trian_writer.add_summary(summaryOut, step)
                if step % 20 == 0:
                    print('Step %d: accuracy = %.2f' % (step, acc_value))
                #
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done traiing, out of data')
        finally:
            coord.request_stop()



        # Now I have to clean up

        coord.join(threads)
        sess.close()



run_training()
