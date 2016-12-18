I need to be able to convert raw png or whatever to tfrecords because they are faster and easier to use... i guess

I have one working file, convertPng.py, which converts my files. In particular I take my .png files, sorted by class (by
directory), and with the class names (nonMNIST_small_labels.txt), I convert.

Files

1) build_image_data.py
    NOT my file, I downloaded this to use as reference
2) convertPng.py
    My file, I made to taek png and turn into TFRecords

Some helpful links have been as follows:

1) Tensorflow read imsages with labels:
    http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

2) How do I convert a directry jepg images to TFRecoreds file in tensorflow
    http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow

3) TF's inception model has a file "build_image_data.py" that builds files under teh assumption that each subdirectory
   represents as lable,
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py

