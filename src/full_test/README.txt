A. Lons
December 2016

This is where I will put together everything I have thus learned and do some example. In this case we will still be
CPU bound. Though I have worked with the GPU part, I will not do that here. That will be the next and last step.

In this example I demonstrate some useful things,
1) A reading pipeline: Setup file-name queue, setup reader/deocder, setup batch with epoch limit (and ways to drop
   out if all files are read), coordinator and queue_runners,
2) Summaries and graph naming

########################################################################################################################
There a number of helpful resources, however some are more important for full instantiation.

1) Tensorflow's fully_connecter_reader.py: this decodes from TFRecord, then setups the ususla batch pipeline, gets the
    Coordinator and queuerunner, and the uses a try/except loop to handle running out of examples. See,
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

2) CIFAR-10
    Not sure where in there they use queue runners and what not, but this has a lot of stuff going on.