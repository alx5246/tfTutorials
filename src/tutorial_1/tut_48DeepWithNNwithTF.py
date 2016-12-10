# Alex Lonsberry
# December 2016
#
# DESCRIPTION
# From sentdex youutube playlist/vidoes "Machine Learning with Python", this is video #48 and is titled : "Processing
# our own Data - Deep Learning with Neural Networks and TensorFlow part 5" and also video #49 and is titles :
# "Preprocessing cont'd - Deep Learning with Neural Networks and TensorFlow part 6"
#
# This file is for preprocessing data, no TF yet.
#
# In this video here we are using our own data, unlike before we were using stricktly the mnist data. Here we are
# going to be using the same network as we were using before. The author uses data set on his website, which are two
# text files (pos.txt and neg.txt which I have both saved in the folder). The files are word data, not simple vectors
# so we need to convert to numerical form and strings are not the same length. We need to handle this.
#
# How is this all going to work
# 1) we create a lexicon (list of all words used)
# 2) we make an array equal to size of lexicon, then we counts in for each word in the sentances (ignore structure)
#
# For this we have to install nltk module (using anaconda of course) for natural language processing..
#
# NOTES
# To get this to work, I originally ran into,
#
#   > Resource 'tokenizers/punkt/english.pickle' not
#   > found.  Please   use the NLTK
#   > Downloader to obtain the resource: >>>
#   > nltk.download().
#
# To fix this I opened another terminal, used anaconda to get to the right interpreture, and then ran python,
# and then ran
#
#   import nltk
#   nltk.download()
#
# which opened a downloader, which I used to download everything, and then it all worked!
# see (http://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load)

import nltk
from nltk.tokenize import word_tokenize # This is for language processing
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

# Lets setup the text file processing stuff, which is not as important as the tensorflow stuff!
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg):
    """
    DESCRIPTION
    This is where we take our two text files and create a lexicon (list of all words represented)
    :param pos:
    :param neg:
    :return:
    """

    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_wods = word_tokenize(l.lower())
                lexicon += list(all_wods)

    # Now that we have all the lexicon, we can lemmitize (truncate words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

    # Drop some words, keep the most telling or relevant words.
    w_counts = Counter(lexicon)  # Creates a dictionary like the following, {'the':52343, 'and':34323 .... }
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2


# Generate some features!
def sample_handling(sample, lexicon, classification):
    """
    DESCRIPTION
    Generate the actual features for each sentence in each textfile document.
    :param sample:
    :param lexicon:
    :param classification:
    :return:
    """

    # We have a list of lists [.... [ [0, 2, 1, 0, 0, ..., 0] [1] ] , .... ] with the bag of words and the class
    featureset = []

    # Open the sample text and parse through the document and generate feastures.
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] = 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=.1):
    """
    DESCRIPTION
    :param pos:
    :param neg:
    :param test_size:
    :return:
    """

    # Create lexicon
    lexicon = create_lexicon(pos, neg)
    print("\nThe length of the lexicon is ", len(lexicon))

    # Create randomly shuffled features set
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    # Make this a numpy.array for tensorflow... This is fucking weird to me, we now have a numpy.array where their are
    # lists at each index
    features = np.array(features)

    #
    testing_size = int(test_size*features.shape[0])

    #
    train_x = list(features[:, 0][:-testing_size]) # The first slicing takes a 2D array and makes 1D, and the second slices that
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    # Run Generation of data set
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt', test_size=.1)

    # Now save dump the saved data.
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


