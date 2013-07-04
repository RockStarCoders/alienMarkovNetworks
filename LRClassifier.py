import random

import numpy as np

import scipy

from cStringIO import StringIO

import skimage.io

import sklearn
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

import joblib

import pomio

import FeatureGenerator

# data IO

def scaleInputData(inputFeatureData):
    # Assumes numeric numpy array [[ data....] , [data....] ... ]
    return preprocessing.scale(inputFeatureData.astype('float'))

def readClassifierFromFile(classifierFileLocation):
    classifier = joblib.load(classifierFileLocation)
    # TODO do a type check type if there is some inheritence/abstraction in sklearn
    return classifier

def splitInputDataset_msrcData(msrcDataLocation, train=0.6, validation=0.2, test = 0.2):
    assert (train + validation + test) == 1, "values for train, validation and test must sum to 1"
    
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to find good regularisation parameters and test for performance
    print "Loading images from msrc dataset"
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    print "Completed loading"
    
    totalImages = np.size(msrcImages)
    
    trainDataSize = np.round(totalImages * train , 0)
    testDataSize = np.round(totalImages * test , 0)
    
    # read data from file, assume msrc data, using pomio
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    
    # Get random samples from list
    trainData, msrcImages = sampleFromList(msrcImages, trainDataSize)
    testData, msrcImages = sampleFromList(msrcImages, testDataSize)
    validationData = msrcImages
    
    print "\nRandomly assigned " + str(np.shape(trainData)) + " subset of msrc data to TRAIN set"
    print "\nRandomly assigned " + str(np.shape(testData)) + " subset of msrc data to TEST set"
    print "\nAssigned remaining " + str(np.shape(validationData)) + " msrc data to VALIDATION set"
    
    return [trainData, validationData, testData]


def reshapeImageLabelData(msrcImage):
    groundTruth = msrcImage.m_gt
    
    numPixels = np.shape(groundTruth)[0] * np.shape(groundTruth)[1]
    return np.reshape(groundTruth, (numPixels))
    

def reshapeImageFeatures(imageFeatures):
    # assume (i, j, f) feature data, so feature array per pixel.  Reshape to (i*j , f) array
    numDatapoints = np.shape(imageFeatures)[0] * np.shape(imageFeatures)[1]
    numFeatures = np.shape(imageFeatures)[2]
    
    return np.reshape(imageFeatures, (numDatapoints, numFeatures))


def sampleFromList(data, numberSamples):
    idx = 0
    result = []
    while idx < numberSamples:
        numImages = np.size(data)
        # randomly sample from imageset, and assign to train
        randIdx = np.round(random.randrange(0,numImages), 0).astype(int)
        result.insert(idx, data[randIdx])
        # now remove the image from the dataset to avoid duplication
        data.pop(randIdx)
        idx = idx+1
        
    return result, data


# Basic classifier functions


def trainLogisticRegressionModel(features, labels, Cvalue, outputClassifierFile, scaleData=True):
    # See [http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html]
    # scaled() method with 1 argument scales data to have zero mean and unit variance
    if scaleData:
        features = preprocessing.scale(features)
    
    # sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    lrc = LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=Cvalue, fit_intercept=True, intercept_scaling=1)
    lrc.fit(features, labels)
    joblib.dump(lrc, outputClassifierFile)
    print "LogisticRegression classifier saved to " + str(outputClassifierFile)
    
    return lrc
    

# Utility functions for train, validation and test for classifier

def testClassifier(classifier, testFeatures, testClassLabels, resultsFile, scaleData=True):
    # predict on testFeatures, compare to testClassLabels, return the rsults
    predictions = classifier.predict(testFeatures)
    
    numberCorrectPredictions = 0
    
    # compare predictions with given class labels
    assert np.shape(predictions) == np.shape(testClassLabels) , "The shape of the prediction: " + str(np.shape(predictions)) + " and given class labels" + str(np.shape(testClassLabels)) + " is not equal!" 
    
    totalCases = np.shape(testClassLabels)[0]
    
    for valueIdx in range(0 , totalCases):
        if predictions[valueIdx] == testClassLabels[valueIdx]:
            numberCorrectPredictions = numberCorrectPredictions + 1
    
    classifierResult = np.append(classifier.getParams(deep=True) , numberCorrectPredictions, totalCases)
    
    # persist the coefficients of the model and the score, so it can be re-created
    np.savetxt(resultsFile, classifierResult, fmt="%s", delimiter=",")
    
    print "Classifier %correct classifications:: ", str( np.round(numberCorrectPredictions / totalCases * 100 , 4))


def processLabelledImageData(inputMsrcImages, outputFileLocation):
    # Assume we get a list / array of msrcImage objects.  We need reshape the labels, and compute+reshape features
    # http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
    
    totalImages = np.size(inputMsrcImages)
    
    totalPixels = 0
    
    for idx in range(0, totalImages):
    
        imageLabels = reshapeImageLabelData(inputMsrcImages[idx])
        
        numPixels = np.size(inputMsrcImages[idx].m_img[:,:,0])
        
        totalPixels = totalPixels + numPixels
        
        print "\nImage#" + str(idx+1) + " has " + str(numPixels) + " pixels"
    
        # TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
        numGradientBins = 9
        
        hog1Darray, hogFeatures = FeatureGenerator.createHistogramOfOrientedGradientFeatures(inputMsrcImages[idx].m_img, numGradientBins, (8,8), (3,3), True, True)
        hog1Darray = None
#         colour3DFeatures = FeatureGenerator.create3dRGBColourHistogramFeature(inputMsrcImages[idx].m_img, 16)
#         colour1DFeatures = FeatureGenerator.create1dRGBColourHistogram(inputMsrcImages[idx].m_img, 16)
        lbpFeatures = FeatureGenerator.createLocalBinaryPatternFeatures(inputMsrcImages[idx].m_img, 4, 2, "default")
        filterResponseFeatures = FeatureGenerator.createFilterbankResponse(inputMsrcImages[idx].m_img, 15)
        
        # resize into (numPixels x numFeatures) array:
        hogFeatures = np.reshape(hogFeatures, (numPixels , np.size(hogFeatures) / numPixels) )
        lbpFeatures = np.reshape(lbpFeatures, (numPixels , np.size(lbpFeatures) / numPixels) )
        filterResponseFeatures = np.reshape(filterResponseFeatures, ( numPixels , np.size(filterResponseFeatures) / numPixels))
        
        imageFeatures = np.hstack( [hogFeatures, lbpFeatures, filterResponseFeatures ] )
        
        # save data to files suitable for classifier by appending to CSV - one for features, one for class labels 
        writeFeaturesToFile(imageFeatures, str(outputFileLocation + "Data.csv"))
        writeFeaturesToFile(imageLabels , str(outputFileLocation + "Labels.csv"))
    
        print "Processed " + str(idx+1) + " of " + str(totalImages) + " images & " + str(totalPixels) + " pixels"


def writeFeaturesToFile(features, filename):
    # http://stackoverflow.com/questions/12218945/formatting-numpy-array
    # read back with np.fromstring(s.getvalue(), sep=',')
    sio = StringIO()
    dataFile = open(filename, 'a')
    
    print "Writing features to file:: " + str(np.shape(features))
    np.savetxt(sio, features, fmt='%.10f', delimiter=',')
    
    dataFile.write(sio.getvalue())
    dataFile.flush()
    
    dataFile.close()

    

def readArrayDataFromFile(arrayDataFile):
    # looks like we need to loop over a read function and construct array, now we are just writing csv to disk
    featureData = None
    with open(arrayDataFile, 'r') as f:
        rowCount = 0;
        row = f.readline()
        while not (row == ''):
            rowArray = np.fromstring(row, sep=',')
            if featureData == None:
                featureData = rowArray
            else:
                featureData = np.vstack( [featureData, rowArray])
        row = f.readline()
        rowCount = rowCount + 1
    return featureData


# TODO look at sklearn pipeline to get some automation here


msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"
trainingPixelFile = "/home/amb/dev/mrf/data/training/pixelLevelData/pixelFeature"
trainingPixelDataFile = trainingPixelFile + "Data.csv"
trainingPixelLabelFile = trainingPixelFile + "Labels.csv"
classifierFile = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/logRegClassifier_C1"


splitData = splitInputDataset_msrcData(msrcData, train=0.6, validation=0.2, test=0.2)
trainData = splitData[0]
validationData = splitData[1]
testData = splitData[2]
processLabelledImageData(trainData, trainingPixelFile)


print "Attempt to load training data from previously generated files..."
print "\t" + trainingPixelDataFile + " & " + trainingPixelLabelFile
data = readArrayDataFromFile(trainingPixelDataFile)
labels  = readArrayDataFromFile(trainingPixelLabelFile)
print "\nNow build logistic regression classifier from input data:" + str(np.shape(trainingPixelDataFile)) + ", " + str(np.shape(trainingPixelLabelFile))
classifier = trainLogisticRegressionModel(data, labels, 1, classifierFile, scaleData=True)
print "Completed classifier training:: " + str(classifier.getParams)



# testData = np.hstack( [np.pi*np.ones((4, 12)) , 2*np.pi*np.ones((4, 3)) , 3*np.pi*np.ones((4,5)) ] )
# testImage = skimage.io.imread("/home/amb/dev/workspaces/mrfWorkspace/objectSegmentation/src/amb/data/ship-at-sea.jpg")
# print testImage
# print "Original test data shape = " + str(np.shape(testImage)) + " , size=" + str(np.size(testImage))
# testData = reshapeImageFeatures(testImage)
# print "\nreshapeImageFeatures result shape = " + str(np.shape(testData)) + " , size=" + str(np.size(testData))
# print testData
# writeFeaturesToFile(testData, "/home/amb/dev/testData.csv")
# # read file
# newArray = None
# print "\nNow try to read int eh file data to nparray:"
# 
# with open('/home/amb/dev/testData.csv', 'r') as f:
#     rowCount = 0;
#     row = f.readline()
#     while not (row == ''):
#         rowArray = np.fromstring(row, sep=',')
#         if newArray == None:
#             newArray = rowArray
#         else:
#             newArray = np.vstack( [newArray, rowArray])
#         
#         row = f.readline()
#         rowCount = rowCount + 1
# print "\nResultant input array shape = " + str(np.shape(newArray)) + ", size=" + str(np.size(newArray))
# make up some random labels, skewed to class=1
# labels = (np.random.random((158000)) < 0.7).astype('int')
# trainLogisticRegressionModel(newArray, labels, 1.0, classifierFile, True)

