A. Lons
December 2016

I am trying to figure out and demonstrate how to use the traditional input pipline, which seems to be based on seperate
threads and queues under the hood.

I have grabbed most my information from the following sources,
    1) TF's How-to on Reading-Data: after reading and looking at forums for help, this stuff finally makes some sense
    https://www.tensorflow.org/how_tos/reading_data/

    2) CIFAR10 example: which reads in
    https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

I have made text file that covers and summarizes much of my understanding. This file is INPUTPIPELINEINFO.txt

Files I have created
    1) readingBinary.py, a demonstraion of a proper pipeline that reads binary data. Here I let tf.train.shuffle_batch()
    handle the multi-threading (fiving the function the number of threads I would like to use)
    2) readingBinaryMultThreads.py, teh same demonstration, expect I by hand make n-thrads of the raeder and pass to
    tf.train.shuffle_batch_join. I think the only point of doing it this way is more shuffle data.

