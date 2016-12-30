A. Lons
December 2016

README for all files in src/tutorial_3/QueueReadingExamples/

The files and examples here are based on generating queues outside of using the more typical pipeline that can be
described here in 'StandardInputPipelineExampeles' or on tensor flows 'Reading data' how-to. Rather, this here is based
more on directly using things like tf.FIFOQueue directly.

########################################################################################################################
Files here,

1) reading_0.py
    This is my file, it is a toy file to try using the basic low-level queus to read through numpy data.

2) reading_1.py
    This is my file, here I use queues in a more distributed form.

3) reading_1_0.py
    This is my file to help generate data for reading_1.py to use.

########################################################################################################################
Some helpful links have been as follows:

1) A great example/examples, and description of the order of operations, is given here in this question and answer. In
   particular, the question was updated based on mmry's anwer and both are very insightful.
    see, http://stackoverflow.com/questions/34594198/how-to-prefetch-data-using-a-custom-python-function-in-tensorflow

2) A good example to describe how make your own pipeline.
    see, https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/