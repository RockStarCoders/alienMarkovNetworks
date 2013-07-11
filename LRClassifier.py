import random

import numpy as np

from cStringIO import StringIO

from skimage import color

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

import joblib

from amb.seg import pomio, FeatureGenerator, PossumStats
from amb.seg.FeatureGenerator import *

#
# Data preparation utils
#

def splitInputDataset_msrcData(msrcDataLocation, datasetScale=1.0 , keepClassDist=True, train=0.6, validation=0.2, test = 0.2):
    assert (train + validation + test) == 1, "values for train, validation and test must sum to 1"
    
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to find good regularisation parameters and test for performance
    print "Loading images from msrc dataset"
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    print "Completed loading"
    
    totalImages = np.size(msrcImages)
    totalSampledImages = np.round(totalImages * datasetScale , 0).astype('int')
    
    trainDataSize = np.round(totalSampledImages * train , 0)
    testDataSize = np.round(totalSampledImages * test , 0)
    validDataSize = np.round(totalSampledImages * validation , 0)
    
    
    # Get random samples from list
    if keepClassDist == False:
        trainData, msrcImages = selectRandomSetFromList(msrcImages, trainDataSize)
        testData, msrcImages = selectRandomSetFromList(msrcImages, testDataSize)
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = selectRandomSetFromList(msrcImages, validDataSize)
            
        print "\nRandomly assigned " + str(np.shape(trainData)) + " subset of msrc data to TRAIN set"
        print "Randomly assigned " + str(np.shape(testData)) + " subset of msrc data to TEST set"
        print "Randomly assigned " + str(np.shape(validationData)) + " msrc data to VALIDATION set"
        
    elif keepClassDist == False:
        # Get class frequency count from image set
        classPixelCount = PossumStats.totalPixelCountPerClass(msrcDataLocation)
        
        # normalise the frequency values to give ratios for each class
        classDist = float(classPixelCount) / float(np.max(classPixelCount))
        
        # use class sample function to create sample sets with same class ratios
        trainData, msrcImages = classSampleFromList(msrcImages, trainDataSize, classDist)
        testData, msrcImages = classSampleFromList(msrcImages, testDataSize, classDist)
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = selectRandomSetFromList(msrcImages, validDataSize)
        
        print "\nAssigned " + str(np.shape(trainData)) + " randomly selected, class ratio preserved samples subset of msrc data to TRAIN set"
        print "Assigned " + str(np.shape(testData)) + " randomly selected, class ratio preserved samples subset of msrc data to TEST set"
        print "Assigned " + str(np.shape(validationData)) + " randomly selected, class ratio preserved samples subset of msrc data to VALIDATION set"
    
    return [trainData, validationData, testData]



def classSampleFromList(data, dataLabels , numberSamples, classDist):
    """This function takes random samples from input dataset (wihout replacement) maintaining a preset class label ratio.
    The indices of the are assumed to align with the indicies of the class labels.
    Returns a list of data samples and the reduced input dataset."""
    
    classSampleSizes = np.round((numberSamples * classDist) , 0).astype('int')
    
    assert (numberSamples == np.sum(classSampleSizes)) , "Some rounding error on total smalpes versus samples by class ::"
    
    sampleResult = []
    
    if np.min(classSampleSizes) == 0:
        print "Check my shizzle!"
        # need to see how many zeros, and either edit the dist or throw an error
    
    # track total samples
    sampleCount = 0
    
    # for each class label    
    for labelIdx in range(0 , np.size(classDist)):
        
        classSampleSize = classSampleSizes[labelIdx]
        
        # reset the value for each class
        classSampleCount = 0
        
        # Get samples and add to list while less than desired sample size for the class
        while classSampleCount < classSampleSize:
            
            # get a random sample from input list
            sample =  randomSampleFromData(data)
            sampleData = sample[0]
            sampleIdx = sample[1]
            
            # check to see if it is the right label
            if sampleIdx == labelIdx:
                
                # if so, add to the sample list and increment counters
                sampleResult = sampleResult.insert( sampleData , np.size(sampleCount) )
                
                data.pop(sampleIdx)
                classSampleCount = classSampleCount + 1
                sampleCount = sampleCount + 1
    
    return sampleResult, data

def selectRandomSetFromList(data, numberSamples):
    """This function randomly selects a number of smaples from an array without replacement.
    Returns an array of the samples, and the resultant reduced data array."""
    idx = 0
    result = []
    while idx < numberSamples:
        # randomly sample from imageset, and assign to sample array
        # randIdx = np.round(random.randrange(0,numImages), 0).astype(int)
        randIdx = np.random.randint(0, np.size(data))
        result.insert(idx, data[randIdx])
        # now remove the image from the dataset to avoid duplication
        data.pop(randIdx)
        idx = idx+1
        
    return result, data

def randomSampleFromData(data):
    """This function selects a member of a data array at random.
    Returns the random sample, and the index of the smaple in the original data array."""
    randIdx = np.random.randint(0, np.size(data))
    
    return [ data[randIdx] , randIdx]


#
# Classifier construction & evaluation utils
#



def trainLogisticRegressionModel(features, labels, Cvalue, outputClassifierFile, scaleData=True):
    # See [http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html]
    # Features are numPixel x unmFeature np arrays, labels are numPixel np array
    
    assert ( np.size( np.shape(labels) ) == 1) , ("Labels should be a 1d array.  Shape of labels = " + str(np.shape(labels)))
    assert (np.size(features[0]) == np.size(labels)) , ("The length of the feature and label data arrays must be equal.  Features=" + str(np.size(features[0])) + ", labels=" + str(np.size(labels)))
    
    if scaleData:
        features = preprocessing.scale(features)
    
    # sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    lrc = LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=Cvalue, fit_intercept=True, intercept_scaling=1)
    lrc.fit(features, labels)
    # See http://stackoverflow.com/questions/10592605/save-naivebayes-classifier-to-disk-in-scikits-learn
    joblib.dump(lrc, outputClassifierFile, compress=1)
    print "LogisticRegression classifier saved to " + str(outputClassifierFile)
    
    return lrc

def testClassifier(classifier, testFeatures, testClassLabels, resultsFile, scaleData=True):
    # predict on testFeatures, compare to testClassLabels, return the results
    predictions = classifier.predict(testFeatures)
    
    numberCorrectPredictions = 0 
    
    totalCases = np.shape(testClassLabels)[0]
    results = None
    
    for valueIdx in range(0 , totalCases):
        
        result = np.array([ predictions[valueIdx] , testClassLabels[valueIdx] ])
        
        # Compile results
        if results == None:
            results = result
        else:
            results = np.vstack((results, result))
        # increment count of correct classifications    
        if predictions[valueIdx] == testClassLabels[valueIdx]:
            numberCorrectPredictions = numberCorrectPredictions + 1
    
    # Save case-by-case scores
    np.savetxt(resultsFile, results, fmt="%.5f", delimiter=",")
    
    # persist summary
    summary = np.array( [ numberCorrectPredictions, totalCases ] )
    sio = StringIO()
    np.savetxt(sio, summary, fmt="%.3f", delimiter=",")
    f = open(resultsFile, 'a')
    f.write(sio.getvalue())
    f.flush()
    f.close()
    
    print "Classifier %correct=:: ", str( np.round(float(numberCorrectPredictions) / float(totalCases) * 100 , 4))
    
def crossValidation_Cparam(trainingData, validationData, classifierBaseFile, classifierBaseTestOutputFile, C_min, C_max, C_increment):
    
    assert (C_min <= C_max) , "C_min param value should be less than or equal to C_max param value."
    assert C_increment == 0, "You specified min and max for C param; C_increment value must NOT be 0"
    assert (np.size(np.shape(trainingData)) == 2 and np.size(np.shape(validationData)) == 2) , ("Training data and test data must be list of length 2 (first element giving feature array, second giving label array. #Training=" + str(np.size(np.shape(trainingData)) + ", #test=" + str(np.shape(validationData)))) 
    
    # Get training data
    trainFeatureData = trainingData[0]
    trainLabelData = trainingData[1]
    assert ((np.size(trainFeatureData[0]) == np.size(trainLabelData))) , "TRAIN data error:: Number of feature vectors must equal number of labels. #trainingFeatures=" + ""
    
    # Get test data
    validationFeatureData = validationData[0]
    validationLabelData = validationData[1]
    assert ((np.size(validationFeatureData[0]) == np.size(validationLabelData))) , "VALIDATION data error:: Number of feature vectors must equal number of labels. #trainingFeatures=" + ""
    
    # Just traing and test on single C_value
    if C_min == C_max:
        print "\nTraining classifier C="+str(C_min)
        classifier = trainLogisticRegressionModel(trainFeatureData, trainLabelData, 0.1, classifierBaseFile + "_" + str(C_min) + "_joblib.pkl", True)
        testClassifier(classifier, validationFeatureData, validationLabelData, classifierBaseTestOutputFile + "_" + str(C_min) + ".csv", True)
    else:
        # Now run cross-validation on each C param, persisting classifier CV performance and corresponding classifier object to file
        for C_param in np.linspace(C_min, C_max, C_increment):
            print "\nTraining classifier C="+str(C_param)
            classifier = trainLogisticRegressionModel(trainFeatureData, trainLabelData, 0.1, classifierBaseFile + "_" + str(C_param) + "_joblib.pkl", True)
            testClassifier(classifier, validationFeatureData, validationLabelData, classifierBaseTestOutputFile + str(C_param) + "_" + ".csv", True)



def processLabeledImageData(inputMsrcImages, outputFileLocation, persistenceType, numGradientBins, numHistBins, ignoreVoid=False,):
# TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
    # Assume we get a list / array of msrcImage objects.  We need reshape the labels, and compute+reshape features
    # http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
    assert (persistenceType == "pickle" or persistenceType == "csv") , "persistenceType must be either pickle or csv"
    
    totalImages = np.size(inputMsrcImages)
    
    allFeatures = None
    allLabels = None
    
    if ignoreVoid == True:
        print "\nVoid class pixels WILL NOT be included in the processed feature dataset"
        
        for idx in range(0, totalImages):
            
            print "\tImage_" , (idx + 1) , " of " , totalImages
            
            imageResult = FeatureGenerator.generateLabeledImageFeatures(inputMsrcImages[idx] , numGradientBins, numHistBins, ignoreVoid=True)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1]
            
            if persistenceType == "csv":
                # Write feature & label data to file, one row vector at a time
                writeFeaturesToCsv(resultFeatures, outputFileLocation)
                writeLabelsToCsv(resultLabels, outputFileLocation)
                
            else:
                # Store results in allData and allLables object for pickle serialization    
                if allFeatures == None:
                    allFeatures = resultFeatures
                else:
                    allFeatures = np.vstack( [ allFeatures, resultFeatures])
             
                if allLabels == None:
                    allLabels = resultLabels
                else:
                    allLabels = np.append( allLabels , resultLabels )
        
    else:
        print "\nVoid class pixels WILL be included in the processed feature dataset"
        for idx in range(0, totalImages):
            imageResult = FeatureGenerator.generateLabeledImageFeatures(inputMsrcImages[idx], numGradientBins, numHistBins, ignoreVoid=False)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1]
            
            if persistenceType == "csv":
                # Write feature & label data to file, one row vector at a time
                writeFeaturesToCsv(resultFeatures, outputFileLocation)
                writeLabelsToCsv(resultLabels, outputFileLocation)
                
            else:
            
                if allFeatures == None:
                    allFeatures = resultFeatures
                else:
                    allFeatures = np.vstack( [ allFeatures, resultFeatures])
             
                if allLabels == None:
                    allLabels = resultLabels
                else:
                    allLabels = np.append( allLabels , resultLabels )
    
    # Now serialize all result data to file
    if persistenceType == "pickle":
        print "\nNow serialising result data to file..."
        persistLabeledDataToFile(allFeatures, allLabels, outputFileLocation, persistenceType)
    



#
# Prediction util methods
#



def generatePredictedPixelClassDist(rgbImage, classifier, numberLabels, numGradientBins, numHistBins):
    """This image taks an RGB image as an (i,j,3) numpy array, a scikit-learn classifier and produces probability distribution over each pixel and class.
    Returns an (i,j,N) numpy array where N= total number of classes for use in subsequent modelling."""
    
    # Take image, generate features, use classifier to predict labels, ensure normalised dist and shape to (i,j,N) np.array
    
    imagePixelFeatures = FeatureGenerator.generatePixelFeaturesForImage(rgbImage, numGradientBins, numHistBins)
    print "Finish me!"


#
# File IO util functions
#


def persistLabeledDataToFile(features, labels, baseFileLocation, persistenceType):
    
    assert (persistenceType == "pickle" or persistenceType == "csv"), "persistenceType must be string value \"pickle\" or \"csv\""
    
    if persistenceType == "csv":
        writeLabeledDataToCSVFile(features, labels, baseFileLocation)
        
    elif persistenceType == "pickle":
        pickleLabeledDataToFile(features, labels, baseFileLocation)


# CSV file utils

def writeLabeledDataToCSVFile(features, labels, baseFileLocation):
    dataFile = str(baseFileLocation + "Data.csv")
    labelFile = str(baseFileLocation + "Labels.csv")
    
    print "Saving image features and labels to::\n\t", dataFile, "\n\t", labelFile
    
    writeFeaturesToCsv(features, baseFileLocation)
    writeLabelsToCsv(labels , baseFileLocation)
 

def writeFeaturesToCsv(features, baseCsvFilename):
    """This function appends a feature array as a new row to an existing CSV file.  If no file exists, one is created."""
    # http://stackoverflow.com/questions/12218945/formatting-numpy-array
    # read back with np.fromstring(s.getvalue(), sep=',')
    featuresFile = str(baseCsvFilename + "Data.csv")
    
    sio = StringIO()
    np.savetxt(sio, features, fmt='%.08f', delimiter=',')
    
    dataFile = open(featuresFile, 'a')
    dataFile.write(sio.getvalue())
    dataFile.flush()
    
    dataFile.close()
    
def writeLabelsToCsv(labels, baseCsvFilename):
    """This function appends an integer class label to an existing CSV file.  If no file exists, one is created."""
    # http://stackoverflow.com/questions/12218945/formatting-numpy-array
    # read back with np.fromstring(s.getvalue(), sep=',')
    labelFile = str(baseCsvFilename + "Labels.csv")
    
    sio = StringIO()
    dataFile = open(labelFile, 'a')
    
    np.savetxt(sio, labels.astype('int'), fmt='%d', delimiter=',')
    
    dataFile.write(sio.getvalue())
    dataFile.flush()
    
    dataFile.close()

def readLabeledDataFromCsv(baseFilename):
    featureData = np.loadtxt(baseFilename + "Data.csv" , dtype='float' , delimiter=',')
    labelData = np.loadtxt(baseFilename + "Labels.csv" , dtype='float' , delimiter=',')
    return [featureData, labelData]


# Pickle serialisation utils

def pickleLabeledDataToFile(features, labels, baseFileLocation):
    pickleNumpyData(features, str(baseFileLocation + "Data.npy"))
    pickleNumpyData(labels , str(baseFileLocation + "Labels.npy"))
    
def pickleNumpyData(data, filename):
    print "Pickle my shizzle"
    np.save(filename, data)
    print "Shizzle is pickled"

def unpickleLabeledData(baseFileLocation):
    features = unpickleNumpyData(str(baseFileLocation + "Data.npy"))
    labels = unpickleNumpyData(str(baseFileLocation + "Labels.npy"))
    return [features, labels]
    
def unpickleNumpyData(filename):
    data = np.load(filename)
    return data


# Classifier IO utils

def loadClassifier(filename):
    return joblib.load(filename)

def readClassifierFromFile(classifierFileLocation):
    classifier = joblib.load(classifierFileLocation)
    # TODO do a type check type if there is some inheritence/abstraction in sklearn
    return classifier

def scaleInputData(inputFeatureData):
    # Assumes numeric numpy array [[ data....] , [data....] ... ]
    return preprocessing.scale(inputFeatureData.astype('float'))


#
# Simple runtime tests
#
# TODO look at sklearn pipeline to get some automation here


msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

# Output file resources
trainingPixelBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/training/trainPixelFeature"

validationPixelBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/crossValidation/data/cvPixelFeature"
validationResultsBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/crossValidation/results/logRegClassifier_CrossValidResult"
 
testingPixelBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/testing/data/testPixelFeature"
testingResultsBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/testing/results/testPixelFeature"
 
classifierBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/classifierModels/logRegClassifier"
 
# Generate data from images and save to file
splitData = splitInputDataset_msrcData(msrcData, datasetScale=0.7, keepClassDist=False, train=0.6, validation=0.2, test=0.2)
trainDataset = splitData[0]
validationDataset = splitData[1]
testDataset = splitData[2]


numGradientBins = 9
numHistBins = 8

# Create & persist feature & label data for training, cross-validation of C apram and classifier testing
# Validation
print "Processing validation data set & writing to CSV::"
processLabeledImageData(validationDataset, validationPixelBaseFilename, "csv", numGradientBins, numHistBins, ignoreVoid=True)
print "Reading persisted validation CSV data to python object::"
validationData = readLabeledDataFromCsv(validationPixelBaseFilename)
print "Serialising validation numpy data object for easier file IO::"
pickleLabeledDataToFile(validationData[0], validationData[1], validationPixelBaseFilename)
print "Initialisation of validation data complete"
validationData = None

# Test
print "Processing validation data set & writing to CSV::"
processLabeledImageData(testDataset, testingPixelBaseFilename, "csv", numGradientBins, numHistBins, ignoreVoid=True)
print "Reading persisted testing CSV data set::"
testingData = readLabeledDataFromCsv(testingPixelBaseFilename)
print "Serialising testing numpy data objects for easier file IO::"
pickleLabeledDataToFile(testingData[0], testingData[1], testingPixelBaseFilename)
print "Initialisation of test data complete."
testingData = None

# Training
print "Processing training dataset & writing to CSV::"
processLabeledImageData(trainDataset, trainingPixelBaseFilename, "csv", numGradientBins, numHistBins, ignoreVoid=True)
print "Reading persisted training CSV data to python object::"
trainingData = readLabeledDataFromCsv(trainingPixelBaseFilename)
print "Serialising training numpy data objects for easier file IO::"
pickleLabeledDataToFile(trainingData[0], trainingData[1], trainingPixelBaseFilename)
print "Initialisation of training data complete."
trainingData = None

# Now load generated data for classification
trainingData = unpickleLabeledData(trainingPixelBaseFilename)
validationData = unpickleLabeledData(validationPixelBaseFilename)

# cross-validation on C param
C_min = 0.1
C_max = 1
C_increment = 0.1
crossValidation_Cparam(trainingData, validationData, classifierBaseFilename, validationResultsBaseFilename, C_min, C_max, C_increment)
