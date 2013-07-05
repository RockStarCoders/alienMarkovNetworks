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
    # See http://stackoverflow.com/questions/10592605/save-naivebayes-classifier-to-disk-in-scikits-learn
    joblib.dump(lrc, outputClassifierFile)
    print "LogisticRegression classifier saved to " + str(outputClassifierFile)
    
    return lrc
    



# Utility functions for train, validation and test for classifier
def testClassifier(classifier, testFeatures, testClassLabels, resultsFile, scaleData=True):
    # predict on testFeatures, compare to testClassLabels, return the rsults
    predictions = classifier.predict(testFeatures)
    
    numberCorrectPredictions = 0 
    
    totalCases = np.shape(testClassLabels)[0]
    
    for valueIdx in range(0 , totalCases):
        if predictions[valueIdx] == testClassLabels[valueIdx]:
            numberCorrectPredictions = numberCorrectPredictions + 1
    
    classifierResult = np.array( [classifier.get_params(deep=True) , numberCorrectPredictions, totalCases ] )
    
    # persist the coefficients of the model and the score, so it can be re-created
    np.savetxt(resultsFile, classifierResult, fmt="%.5f", delimiter=",")
    
    print "Classifier %correct classifications:: ", str( np.round(numberCorrectPredictions / totalCases * 100 , 4))


def processLabelledImageData(inputMsrcImages, outputFileLocation, persistenceType):
    # Assume we get a list / array of msrcImage objects.  We need reshape the labels, and compute+reshape features
    # http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
    assert (persistenceType == "pickle" or persistenceType == "csv") , "persistenceType must be either pickle or csv"
    
    totalImages = np.size(inputMsrcImages)
    totalPixels = 0
    
    if(persistenceType == "csv"):
        
        for idx in range(0, totalImages):
    
            imageLabels = reshapeImageLabelData(inputMsrcImages[idx])
            
            numPixels = np.size(inputMsrcImages[idx].m_img[:,:,0])
            
            totalPixels = totalPixels + numPixels
            
            print "\nImage#" + str(idx+1) + " has " + str(numPixels) + " pixels"
        
            # TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
            imageFeatures = generateFeaturesForImage(inputMsrcImages[idx])
            
            # save data to files suitable for classifier by appending to CSV - one for features, one for class labels 
            writeFeaturesToFile(imageFeatures, str(outputFileLocation + "Data.csv"))
            writeFeaturesToFile(imageLabels , str(outputFileLocation + "Labels.csv"))
            
            print "Processed " + str(idx+1) + " of " + str(totalImages) + " images & " + str(totalPixels) + " pixels"
            
    elif(persistenceType == "pickle"):
        print "In pickle mode, accumulate feature data in memeory, then serialize to file"
        
        allFeatures = None
        allLabels = None
        
        for idx in range(0, totalImages):
            
            imageLabels = reshapeImageLabelData(inputMsrcImages[idx])
            
            numPixels = np.size(inputMsrcImages[idx].m_img[:,:,0])
            totalPixels = totalPixels + numPixels
            
            print "\nImage#" + str(idx+1) + " has " + str(numPixels) + " pixels"
        
            # TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
            imageFeatures = generateFeaturesForImage(inputMsrcImages[idx])
            
            if allFeatures == None:
                allFeatures = imageFeatures
            else:
                allFeatures = np.vstack( [ allFeatures, imageFeatures])
            
            if allLabels == None:
                allLabels = imageLabels
            else:
                allLabels = np.append( allLabels , imageLabels )
            
            # save data to files suitable for classifier by appending to CSV - one for features, one for class labels
            print "Processed " + str(idx+1) + " of " + str(totalImages) + " images & " + str(totalPixels) + " pixels"
            
        # Now serialize entire arrays to file 
        pickleNumpyData(imageFeatures, str(outputFileLocation + "Data.npy"))
        pickleNumpyData(imageLabels , str(outputFileLocation + "Labels.npy"))
        

def generateFeaturesForImage(msrcImage):
    
    numGradientBins = 9
    
    numPixels = np.size(msrcImage.m_img[:,:,0])
        
    hog1Darray, hogFeatures = FeatureGenerator.createHistogramOfOrientedGradientFeatures(msrcImage.m_img, numGradientBins, (8,8), (3,3), True, True)
    hog1Darray = None
#     colour3DFeatures = FeatureGenerator.create3dRGBColourHistogramFeature(msrcImage.m_img, 16)
#     colour1DFeatures = FeatureGenerator.create1dRGBColourHistogram(msrcImage.m_img, 16)
    lbpFeatures = FeatureGenerator.createLocalBinaryPatternFeatures(msrcImage.m_img, 4, 2, "default")
    filterResponseFeatures = FeatureGenerator.createFilterbankResponse(msrcImage.m_img, 15)
        
    # resize into (numPixels x numFeatures) array:
    hogFeatures = np.reshape(hogFeatures, (numPixels , np.size(hogFeatures) / numPixels) )
    lbpFeatures = np.reshape(lbpFeatures, (numPixels , np.size(lbpFeatures) / numPixels) )
    filterResponseFeatures = np.reshape(filterResponseFeatures, ( numPixels , np.size(filterResponseFeatures) / numPixels))
        
    imageFeatures = np.hstack( [hogFeatures, lbpFeatures, filterResponseFeatures ] )
        
    return imageFeatures




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


def pickleNumpyData(data, filename):
    print "Pickle my shizzle"
    np.save(filename, data)
    print "Shizzle is now pickled"


def unpickleNumpyData(filename):
    data = np.load(filename)
    return data

    
def readArrayDataFromFile(arrayDataFile):
    # looks like we need to loop over a read function and construct array, now we are just writing csv to disk
    featureData = None
    print "\nReading data from file::" + str(arrayDataFile)
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
classifierFile = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/logRegClassifier_C1_joblib.pkl"

# Generate data from images and save to file
# splitData = splitInputDataset_msrcData(msrcData, train=0.6, validation=0.2, test=0.2)
# trainData = splitData[0]
# validationData = splitData[1]
# testData = splitData[2]
# processLabelledImageData(trainData, trainingPixelFile)


# print "Attempt to load training data from previously generated files..."
# print "\t" + trainingPixelDataFile + " & " + trainingPixelLabelFile
# print "First the class label data..."
# labels  = readArrayDataFromFile(trainingPixelLabelFile)
# print "Now the feature data..."
# data = readArrayDataFromFile(trainingPixelDataFile)
# print "\nNow build logistic regression classifier from input data:" + str(np.shape(trainingPixelDataFile)) + ", " + str(np.shape(trainingPixelLabelFile))
# classifier = trainLogisticRegressionModel(data, labels, 1, classifierFile, scaleData=True)
# print "Completed classifier training:: " + str(classifier.getParams)


# toy example with small number of training images
toyTrainPixelFile = "/home/amb/dev/mrf/data/training/pixelLevelData/toyPixelFeature"
toyTestPixelFile = "/home/amb/dev/mrf/data/test/pixelLevelData/toyPixelFeature"

# toyTrainPixelFeaturesCsv = toyTrainPixelFile + "Data"
# toyTrainPixelLabelsCsv = toyTrainPixelFile + "Labels"
# toyTestPixelFeaturesCsv = toyTestPixelFile + "Data"
# toyTestPixelLabelsCsv = toyTestPixelFile + "Labels"

toyTrainPixelFeaturesPickle = toyTrainPixelFile + "Data.npy"
toyTrainPixelLabelsPickle = toyTrainPixelFile + "Labels.npy"
toyTestPixelFeaturesPickle = toyTestPixelFile + "Data.npy"
toyTestPixelLabelsPickle = toyTestPixelFile + "Labels.npy"

toyClassifierFile = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/toyLogRegClassifier_C1_joblib.pkl"

testResultsFile = "/home/amb/dev/mrf/data/test/pixelLevelData/toyClassifier_testResults.csv"

msrcData = pomio.msrc_loadImages(msrcData)
toyTrainData, reducedData = sampleFromList(msrcData, 2)
toyTestData, reducedData = sampleFromList(reducedData, 1)

# Generate training data
processLabelledImageData(toyTrainData, toyTrainPixelFile , "pickle")

trainFeatureData = unpickleNumpyData(toyTrainPixelFeaturesPickle)
trainLabelData = unpickleNumpyData(toyTrainPixelLabelsPickle)

print "\n***Now training classifier..."
toyClassifier = trainLogisticRegressionModel(trainFeatureData, trainLabelData, 1.0, toyClassifierFile, True)

# Generate testing data (pickle or csv)
processLabelledImageData(toyTestData, toyTestPixelFile , "pickle")

testFeatureData = unpickleNumpyData(toyTestPixelFeaturesPickle)
testLabelData = readArrayDataFromFile(toyTestPixelLabelsPickle)

print "\n***Now testing classifier..."
testClassifier(toyClassifier, testFeatureData, testLabelData, testResultsFile, True)


