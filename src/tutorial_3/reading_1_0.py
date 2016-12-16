#
#
# DESCRIPTION
# Just a support file to basically create and save data.

import numpy as np
import pickle


# We want to store each numpy array in a seperate file, so we need a list of file names!
numb_files = 100
for i in range(numb_files):
    # Create a numpy object
    inputData = np.ones([5])*i
    outputData = np.ones([3])*i*2
    # Create file names
    outStr = str(i)
    outStr = outStr.zfill(4)
    outStr = "npDataFiles/"+outStr
    # Pickle the shit
    f = open(outStr, 'wb')
    pickle.dump(inputData, f)
    pickle.dump(outputData, f)
    f.close()

from os import listdir
print(listdir("npDataFiles"))
listOfFiles = listdir("npDataFiles")
f = open("npDataFiles/"+listOfFiles[0], "rb")
inputData = pickle.load(f)
print(len(listOfFiles))
print(inputData)