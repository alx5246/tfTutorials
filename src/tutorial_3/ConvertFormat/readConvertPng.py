# A. Lons
# Deceber 2016
#
# Once I have converted data to the TFRecord format I need to be able to read it!. Here I am relying on the code
# from tensorflow/examples/how_tos/reading_date/fully_connected_reader.py
#


import tensorflow as tf

# Not yet sure how to set this?
NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

def read_and_decode_TFR(filename_queue):
    """
    This is a modified version of read_and_decode from ...fully_connected_reader.py. NOTE: here I am assigning image
    size on the fly, which is not good for tensorflow to build graph so I need to handle elsewhere!
    :param filename_queue:
    :return:
    """

    # I use the reader specific to TFRecrods
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # (AJL) from orig file
    #features = tf.parse_single_example(serialized_example,
    #                                    # Defaults are not specified since both keys are required.
    #                                    features={
    #                                        'image_raw': tf.FixedLenFeature([], tf.string),
    #                                        'label': tf.FixedLenFeature([], tf.int64),
    #                                    })

    features = tf.parse_single_example(serialized_example, features={'image/height': tf.FixedLenFeature([], tf.int64),
                                                                     'image/width': tf.FixedLenFeature([], tf.int64),
                                                                     'image/colorspace': tf.FixedLenFeature([], tf.string),
                                                                     'image/channels': tf.FixedLenFeature([], tf.int64),
                                                                     'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                                     'image/class/text': tf.FixedLenFeature([], tf.string),
                                                                     'image/format': tf.FixedLenFeature([], tf.string),
                                                                     'image/filename': tf.FixedLenFeature([], tf.string),
                                                                     'image/encoded': tf.FixedLenFeature([], tf.string)} )

    # The image/encodes was originally stored as string, we have to decode this into a the jpeg
    #image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    i_height = tf.cast(features['image/height'], tf.int64)
    i_width = tf.cast(features['image/width'], tf.int64)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    print(type(image))
    print(tf.shape(image))

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    depth = 3
    image = tf.reshape(image, [height, width, depth])

    print(type(image))
    print(tf.shape(image))


    # (AJL) from orig file
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    #image = tf.decode_raw(features['image_raw'], tf.uint8)
    #image.set_shape([mnist.IMAGE_PIXELS])
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int64)

    return image, label


def input_pipline(filenames, batch_size, numb_pre_threads):
    """
    DESCRIPTION
        In accordance with your typical pipelien that I have denoted, we have a seperate method that sets up the
        data.
    :param filenames:
    :param batch_size:
    :return:
    """

    # Generate the file-name queue from given list of filenames
    filename_queue = tf.train.string_input_producer(filenames)
    # Read the image using method defined above
    image, label = read_and_decode_TFR(filename_queue)

    # Use tf.train.shuffle_batch to shuffle up batches. "min_after_dequeue" defines how big a buffer we will randomly
    # sample from -- bigger means better shuffling but slower start up and more memory used. "capacity" must be larger
    # than "min_after_dequeue" and the amount larger determines the maximm we will prefetch. The recommendation:
    # for capacity is min_after_dequeue + (num_threads + saftey factor) * batch_size
    # From cifar10_input.input(), setup min numb of exampels in teh queue
    min_fraction_of_examples_in_queue = .6
    min_queue_examples = int(NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    min_after_dequeue = min_queue_examples
    print(min_after_dequeue)
    capacity = min_queue_examples + 3 * batch_size
    print(capacity)
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=numb_pre_threads,
                                                 capacity=capacity,
                                                 min_after_dequeue=min_after_dequeue)



    return images, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':

    #Here we will run the test! This will test our abilities to set everything correctly!

    # Get file names
    filenames = ['notMNIST_conv/notMNISTdata-00000-of-00128',
                 'notMNIST_conv/notMNISTdata-00001-of-00128',
                 'notMNIST_conv/notMNISTdata-00003-of-00128',
                 'notMNIST_conv/notMNISTdata-00004-of-00128',
                 'notMNIST_conv/notMNISTdata-00005-of-00128',
                 'notMNIST_conv/notMNISTdata-00006-of-00128',
                 'notMNIST_conv/notMNISTdata-00007-of-00128',
                 'notMNIST_conv/notMNISTdata-00008-of-00128',
                 'notMNIST_conv/notMNISTdata-00009-of-00128']

    images, labels = input_pipline(filenames, batch_size=10, numb_pre_threads=1)

    # This is done in one how-to example and in cafir-10 example.
    init_op = tf.global_variables_initializer()

    # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs)
    sess = tf.Session()

    # Run the init, this is done in how-to and cifar-10
    sess.run(init_op)

    # Make a coordinator,
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100):

        a, b = sess.run([images, labels])
        print("\n")
        print(a.shape)
        print(b)



    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()