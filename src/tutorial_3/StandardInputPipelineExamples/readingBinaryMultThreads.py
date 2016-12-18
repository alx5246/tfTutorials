# A.Lonsberry
# December 2016
#
# In most of the examples I find for tensorflow, they read in binary data. For example, the CIFAR10 reads in binary
# data as well. So what I am doing here is mostly copying from the CIFAR10 example reader:
# tensorflow/models/tutorials/image/cifar10/cifar_10.py, and looking at the how-to on Reading data.
#
# The files and guides that helped me here,
# 1) File that reads binary data: tensorflow/models/tutorials/image/cifar10/cifar_10.py
# 2) A how-to by TF: https://www.tensorflow.org/how_tos/reading_data/
# 3) Reading binary with TF example: http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
#
# We are have a complete pipeline for binary data, unlike readingBinary.py, I will also include MULTIPLE
# read_binary_image() calls. However from what I understand from TF's how-to on "reading data", the only real difference
# between using tf.train.shuffle_batch() and setting n-threads, and tf.train.suffle_bath_join() where the input is a
# list of multple threads is how shuffled the data gets. The later apparently ahuffles the data more.

import tensorflow as tf

NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN = 32

def read_binary_image(filename_queue):
    """
    DESCRIPTION
    This is originally taken from teh cifar-10 example (cifar10_input.read_cifar10()), and then modified for my purposes.
    It should work very in a similar way. I changed this to handle parrallel calls.

    From the original file "Reads and parses examples from CIFAR10 data files."

    ARGS:
        a filename_queue: A queue of strings with the filesnames to read from.
    RETURNS:
        An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
        for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    print('enter: read_binary_image() method')

    # (AJL) make a dummy class? They do this in the examples...
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset, input format. The following are done to find the size of the
    # files!
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting file names from the filename_queue. Readers are tensorflows way for reading data formats.
    # No header or footer in the CIFAR-10 format, so we leave header_bytes and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # (AJL) the "reader" will dequeue a work unit from teh queue if necessary (e.g. when the Reader needs to start
    # reading from a new file since it has finished with the previous file)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape from [depth * height * width] to [depth,
    # height, width]. I change from tf.strided_slice(), because it was not working, and instad went to tf.slice so I
    # also had to change the values as they are different.
    inter = tf.slice(record_bytes, [label_bytes], [image_bytes])
    depth_major = tf.reshape(inter, [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    outImage = tf.cast(result.uint8image, tf.float32)
    outLable = result.label

    return outImage, outLable


def input_pipline(filenames, batch_size, numb_pre_threads):
    """
    DESCRIPTION
        In accordance with your typical pipelien that I have denoted, we have a seperate method that sets up the
        data. Here I will use multiple threads
    :param filenames:
    :param batch_size:
    :return:
    """

    # Generate the file-name queue from given list of filenames
    filename_queue = tf.train.string_input_producer(filenames)

    # Read the image using method defined above, this time in multiple threads! We also have to to handle the casting
    # in the read function!
    #read_input = read_binary_image(filename_queue)
    # I dumped the reshape into the thing above
    #reshape_image = tf.cast(read_input.uint8image, tf.float32)
    inputReadsList = [read_binary_image(filename_queue) for _ in range(4)]

    # Use tf.train.shuffle_batch to shuffle up batches. "min_after_dequeue" defines how big a buffer we will randomly
    # sample from -- bigger means better shuffling but slower start up and more memory used. "capacity" must be larger
    # than "min_after_dequeue" and the amount larger determines the maximm we will prefetch. The recommendation:
    # for capacity is min_after_dequeue + (num_threads + saftey factor) * batch_size
    # From cifar10_input.input(), setup min numb of exampels in teh queue
    min_fraction_of_examples_in_queue = .4
    min_queue_examples = int(NUMB_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    min_after_dequeue = min_queue_examples
    capacity = min_queue_examples + 3 * batch_size
    # I had to change this for multiple input threads
    #images, label_batch, key = tf.train.shuffle_batch([reshape_image, read_input.label, read_input.key],
    #                                                  batch_size=batch_size, num_threads=numb_pre_threads,
    #                                                  capacity=capacity,
    #                                                  min_after_dequeue=min_after_dequeue)
    image_batch, label_batch = tf.train.shuffle_batch_join(inputReadsList,
                                                           batch_size=batch_size,
                                                           capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue)

    return image_batch, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':

    #Here we will run the test! This will test our abilities to set everything correctly!

    # Get file names
    filenames = ['cifar-10-batches-bin/data_batch_1.bin']

    images, labels = input_pipline(filenames, batch_size=4, numb_pre_threads=2)

    # This is done in one how-to example and in cafir-10 example.
    init_op = tf.global_variables_initializer()

    # Create a session, this is done in how-to and cifar-10 example (in the cifar-10 the also have some configs)
    sess = tf.Session()

    # Run the init, this is done in how-to and cifar-10
    sess.run(init_op)

    # Make a coordinator,
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):

        a, b = sess.run([images, labels])
        print("\n")
        print(type(a))
        print(b)


        #print(labels)


    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()