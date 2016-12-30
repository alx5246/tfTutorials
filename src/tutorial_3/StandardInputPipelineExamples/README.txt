A. Lons
December 2016

README for all files in src/tutorial_3/StandardInputPipelineExamples/

I am trying to figure out and demonstrate how to use the traditional input-pipline, which seems to be based on seperate
threads and queues under the hood.

########################################################################################################################
Files here,

1) I have made text file that covers and summarizes much of my understanding. This file is INPUTPIPELINEINFO.tx

2) readingBinary.py, a demonstraion of a proper pipeline that reads binary data. Here I let tf.train.shuffle_batch()
    handle the multi-threading (fiving the function the number of threads I would like to use)

3) readingBinaryMultThreads.py, teh same demonstration, expect I by hand make n-thrads of the raeder and pass to
    tf.train.shuffle_batch_join. I think the only point of doing it this way is more shuffle data.

########################################################################################################################
Some helpful links have been as follows:

1) TF's How-to on Reading-Data: after reading and looking at forums for help, this stuff finally makes some sense
    https://www.tensorflow.org/how_tos/reading_data/

2) CIFAR10 example: which reads in binary data
    https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

3) Another good example is "fully-connected_reader.py" where they read in, create a batch, and run coordinator and
    start the threads up. This is perhaps one of the most straight-forward examples.
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

4) Readingbinary with TF example:
    http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display





