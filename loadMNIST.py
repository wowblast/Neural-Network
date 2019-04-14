import numpy as np

def getTrainData():
    return getData("mnist_train.npz")

def getTestData():
    return getData("mnist_test.npz")

def getValidationData():
    return getData("mnist_valid.npz")

def getData(fname):
    with np.load(fname) as data:
        images = data['images']
        labels = data['labels']
    return images, labels

print (getTestData()[1][0])
print (getTestData()[0][0])
#print len(getTestData()[0][0])

#print len(getTrainData()[0])