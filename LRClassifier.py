import random

import numpy as np

import scipy

import skimage

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
    
    numImages = np.size(inputMsrcImages)
    
    allLabels = None
    
    for labelIdx in range(0, numImages):
        # stack up the labels
        labels = reshapeImageLabelData(inputMsrcImages[labelIdx])
        if allLabels == None:
            allLabels = labels
        else:
            allLabels = np.hstack( [ allLabels, labels ] )
    
    allFeatures = None
    
    for idx in range(0, numImages):
    
        numPixels = np.size(inputMsrcImages[idx].m_img[:,:,0])
        
        print "\nImage#" + str(idx+1) + " has " + str(numPixels) + " pixels"
    
        # TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
        numGradientBins = 9
        hog1Darray, hogFeatures = FeatureGenerator.createHistogramOfOrientedGradientFeatures(inputMsrcImages[idx].m_img, numGradientBins, (8,8), (3,3), True, True)
#         colour3DFeatures = FeatureGenerator.create3dRGBColourHistogramFeature(inputMsrcImages[idx].m_img, 16)
#         colour1DFeatures = FeatureGenerator.create1dRGBColourHistogram(inputMsrcImages[idx].m_img, 16)
        lbpFeatures = FeatureGenerator.createLocalBinaryPatternFeatures(inputMsrcImages[idx].m_img, 4, 2, "default")
        filterResponseFeatures = FeatureGenerator.createFilterbankResponse(inputMsrcImages[idx].m_img, 15)
        
        # resize into (numPixels x numFeatures) array:
        hogFeatures = np.reshape(hogFeatures, (numPixels , np.size(hogFeatures) / numPixels) )
        lbpFeatures = np.reshape(lbpFeatures, (numPixels , np.size(lbpFeatures) / numPixels) )
        filterResponseFeatures = np.reshape(filterResponseFeatures, ( numPixels , np.size(filterResponseFeatures) / numPixels))
#         print "Image feature sizes:"
#         print "\tHOG: " + str(np.size(hogFeatures)) + " , " + str(np.shape(hogFeatures))
#         print "\t3d colour histogram (16 bins): " + str(np.size(colour3DFeatures)) + " , " + str(np.shape(colour3DFeatures))
#         print "\t1d colour histogram (16 bins): " + str(np.size(colour1DFeatures)) + " , " + str(np.shape(colour1DFeatures))
#         print "\tLBP features (4 neighbour): " + str(np.size(lbpFeatures)) + " , " + str(np.shape(lbpFeatures))
#         print "\tfilter response features (15x15 window): " + str(np.size(filterResponseFeatures)) + " , " + str(np.shape(filterResponseFeatures))
    
        imageFeatures = np.hstack( [hogFeatures, lbpFeatures, filterResponseFeatures ] )
        print "Image features array:: " + str(np.size(imageFeatures))
    
        if allFeatures == None:
            allFeatures = imageFeatures
        else:
            np.vstack( [allFeatures, imageFeatures] )
    
    # save data to file
    result = np.array( [ allFeatures, allLabels ])
    
    np.savetxt(str(outputFileLocation + "Data.csv"), result[0], delimiter=",", fmt="%s")
    np.savetxt(str(outputFileLocation + "Labels.csv"), result[1], delimiter=",", fmt="%s")
    
    # return results
    return result
    

# TODO look at sklearn pipeline to get some automation here

msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"
trainingPixelFeatureDataFile = "/home/amb/dev/mrf/data/training/pixelLevelData/pixelFeature"
classifierFile = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/logRegClassifier_C1"

splitData = splitInputDataset_msrcData(msrcData, train=0.6, validation=0.2, test=0.2)

trainData = splitData[0]
# validationData = splitData[1]
# testData = splitData[2]

result = processLabelledImageData(trainData, trainingPixelFeatureDataFile)

classifier = trainLogisticRegressionModel(result[0], result[1], 1, classifierFile, scaleData=True)
    
