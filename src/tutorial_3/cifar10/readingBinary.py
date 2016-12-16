# A.Lonsberry
# December 2016
#
# In most of the examples I find for tensorflow, they read in binary data. For example, the CIFAR10 reads in binary
# data as well. So what I am doing here is mostly compying the CIFAR10 example reader.
#
# We are going to test these out at the bottom

import tensorflow as tf

def read_binary_image(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    ARGS: a filename
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

    print('enter read_binary_image')

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

    # Read a record, getting file names from the filename_queue. No header or footer in the CIFAR-10 format, so we
    # leave header_bytes and footer_bytes at their default of 0.
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

    return result


def input_pipline(filenames, batch_size):

    # Generate the file-name queue
    filename_queue = tf.train.string_input_producer(filenames)

    # Read the image
    read_input = read_binary_image(filename_queue)

    # This will be the actual image
    reshape_image = tf.cast(read_input.uint8image, tf.float32)



    #
    images, label_batch = tf.train.shuffle_batch([reshape_image, read_input.label], batch_size=batch_size, capacity=100,
                                                 min_after_dequeue=6)

    return images, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':

    #Here we will run the test! This will test our abilities to set everything correctly!

    # Get file names
    filenames = ['cifar-10-batches-bin/data_batch_1.bin']

    images, labels = input_pipline(filenames, batch_size=4)

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
        print(b)

        #print(labels)


    # Now I have to clean up
    coord.request_stop()
    coord.join(threads)
    sess.close()


