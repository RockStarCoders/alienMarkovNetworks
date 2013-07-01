import random

import numpy as np

import scipy

import skimage

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import joblib

import pomio

# data IO

def cleanInputData(inputFeatureData):
    # Assumes numeric numpy array [[ data....] , [data....] ... ]
    return preprocessing.Scaler().transform(inputFeatureData)

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
    print "\nAssigned remaining" + str(np.shape(validationData)) + " msrc data to validation set"
    
    # Now reshape and format to np arrays 
    trainData, trainLabels = convertImagesToData(trainData)
    testData, testLabels = convertImagesToData(testData)
    validationData, validationLabels = convertImagesToData(validationData)
    
    return [trainData, trainLabels, validationData, validationLabels, testData, testLabels]
    

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
        print "\tData assignment reduced image set size to: " + str(np.size(data))
        idx = idx+1
        
    return result, data


def convertImagesToData(msrcImages):
    dataResult = None
    labelsResult = None
    
    for idx in range(0, np.size(msrcImages)):
        data, labels = convertMsrcImageToInputData(msrcImages[idx])
        
        if dataResult == None:
            dataResult = data
        else:
            print "\t*How do data results stack up?  :: " + str(np.shape(dataResult)) + "" + str(np.shape(data))
            dataResult = np.vstack([dataResult, data])
            
        if labelsResult == None:
            labelsResult = labels
        else:
            labelsResult = np.vstack([labelsResult, labels])
            
    return dataResult, labelsResult



def convertMsrcImageToInputData(msrcImage):
    # we get numpy arrays from pomio, so just reshape
    data = msrcImage.m_img
    labels = msrcImage.m_gt
    
    assert ((data.shape[0] == labels.shape[0]) and (data.shape[1] == labels.shape[1])), "The dimensions of the ground truth " +\
                                str(labels.shape)+ "and image data " +\
                                str(data.shape) + " from the msrcImage object do not match!"
    
    # make a 1-dimensional view of arr
    data = data.ravel()
    labels = labels.ravel()
    
    # FIXME need to cope with different sized images - use some resize function to scale up to maximum picture size?
    return data , labels


# Basic classifier functions

def trainLogisticRegressionModel(features, labels, C, outputClassifierFile, scaleData=True):
    # See [http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html]
    # scaled() method with 1 argument scales data to have zero mean and unit variance
    if scaleData:
        features = preprocessing.Scaler().transform(features)
    
    lrc = LogisticRegression(C, dual=False, fit_intercept=True, sintercept_scaling=1, penalty='l1', tol=0.0001)
    lrc.fit(features, labels)
    joblib.dump(lrc, outputClassifierFile)
    print "LogisticRegression classifier saved to " + str(outputClassifierFile)


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



# TODO look at sklean pipeline to get some automation here

msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

splitInputDataset_msrcData(msrcData, train=0.6, validation=0.2, test=0.2)
    
