A. Lons
December 2016

README for all files in src/tutorial_3/ConverFormat/

Here I am demonstrating how to 1) alter files to TFRecords format, and then 2) how to load those back in using
the 'standard' type reading pipeline.

########################################################################################################################
Files here,

1) build_image_data.py
    NOT my file, I downloaded this to use as reference for my own code.

2) convertPng.py
    My file, I made to take a .png and turn into TFRecords. In particular, I use the files in notMNIST_small/, which
    is a directory filled with other folders. Each sub-folder, for example, "notMNIST_small/A/", is named with a class
    label. That is, all the files in "notMNIST_small/A/" are of class "A". This converPng.py file has the methods to
    (a) read all the files, and generate the list of files, classes, and labels. There are methods there then to take
    these, and convert to TRFecords, which are then placed in the directory notMNIST_cov/. N

3) readConverPng.py
    My file which has methods to take a list of files with TFRecords and open them. In this file I
    demonstrate how to open the files saved in notMNIST_conv/.

4) notMBIST_small_labels.txt
    A text file with labels, I need this for covertPng.py methods as it tells me what the sub-folders, in nonMNIST_small
    to look for in terms of class labels.

########################################################################################################################
Some helpful links have been as follows:

1) Tensorflow read-images with labels, this was a bit helpful.
    http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

2) How do I convert a directry jepg images to TFRecoreds file in tensorflow. This was a very helpful resource for me
   in getting my own code to work.
    http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow

3) TF's inception model has a file "build_image_data.py" that builds files under teh assumption that each subdirectory
   represents as label,
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py

4) TFR record reading
    http://stackoverflow.com/questions/35028173/how-to-read-images-with-different-size-in-a-tfrecord-file

5) Tensorflow's fully_connected_read.py: In here we have a full pipeline where we read-in and decode the TFRecrod type,
    then setup the pipeline using the usual batch stuff, and then run a training op using a Coordinator(), queuerunner,
    and a try: statement to handle running out of examples!

6) I copied the following for use in 'readConvertPng.py': tensorflow/examples/how_tos/reading_date/fully_connected_reader.py

7) Reshaping loaded image from TFR
    https://github.com/tensorflow/tensorflow/issues/2604


