import random

import numpy as np

from cStringIO import StringIO

from skimage import color

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

import joblib

import pomio, FeatureGenerator, PossumStats



#
# Data preparation utils
#

def splitInputDataset_msrcData(msrcDataLocation, keepClassDist=True, train=0.6, validation=0.2, test = 0.2):
    assert (train + validation + test) == 1, "values for train, validation and test must sum to 1"
    
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to find good regularisation parameters and test for performance
    print "Loading images from msrc dataset"
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    print "Completed loading"
    
    totalImages = np.size(msrcImages)
    
    trainDataSize = np.round(totalImages * train , 0)
    testDataSize = np.round(totalImages * test , 0)
    # validationData is the remainder from train and test
    
    
    # Get random samples from list
    if keepClassDist == False:
        trainData, msrcImages = selectRandomSetFromList(msrcImages, trainDataSize)
        testData, msrcImages = selectRandomSetFromList(msrcImages, testDataSize)
        validationData = msrcImages
        
        print "\nRandomly assigned " + str(np.shape(trainData)) + " subset of msrc data to TRAIN set"
        print "Randomly assigned " + str(np.shape(testData)) + " subset of msrc data to TEST set"
        print "Assigned remaining " + str(np.shape(validationData)) + " msrc data to VALIDATION set"
        
    elif keepClassDist == False:
        # Get class frequency count from image set
        classPixelCount = PossumStats.totalPixelCountPerClass(msrcDataLocation)
        
        # normalise the frequency values to give ratios for each class
        classDist = float(classPixelCount) / float(np.max(classPixelCount))
        
        # use class sample function to create sample sets with same class ratios
        trainData, msrcImages = classSampleFromList(msrcImages, trainDataSize, classDist)
        testData, msrcImages = classSampleFromList(msrcImages, testDataSize, classDist)
        validationData = msrcImages
        
        print "\nAssigned " + str(np.shape(trainData)) + " randomly selected, class ratio preserved samples subset of msrc data to TRAIN set"
        print "Assigned " + str(np.shape(testData)) + " randomly selected, class ratio preserved samples subset of msrc data to TEST set"
        print "Assigned " + str(np.shape(validationData)) + " randomly selected, class ratio preserved samples subset of msrc data to VALIDATION set"
    
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


def processLabeledImageData(inputMsrcImages, outputFileLocation, persistenceType, ignoreVoid=False):
    # Assume we get a list / array of msrcImage objects.  We need reshape the labels, and compute+reshape features
    # http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
    assert (persistenceType == "pickle" or persistenceType == "csv") , "persistenceType must be either pickle or csv"
    
    totalImages = np.size(inputMsrcImages)
    
    allFeatures = None
    allLabels = None
    
    if ignoreVoid == True:
        print "\nVoid class pixels WILL NOT be included in the processed feature dataset"
        
        for idx in range(0, totalImages):
            print "Processing image " , (idx + 1)
            # TODO refactor this into FeatureGenerator.py as a util method (you give me image, I give you image feature data over pixels)
            imageResult = generateLabeledImageFeatures(inputMsrcImages[idx], ignoreVoid=True)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1] 
            
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
            imageResult = generateLabeledImageFeatures(inputMsrcImages[idx].m_img, ignoreVoid=False)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1]
            if allFeatures == None:
                allFeatures = resultFeatures
            else:
                allFeatures = np.vstack( [ allFeatures, resultFeatures])
             
            if allLabels == None:
                allLabels = resultLabels
            else:
                allLabels = np.append( allLabels , resultLabels )
    
    # Now save data to file
    print "\nNow persisting result data to file..."
    persistLabeledDataToFile(allFeatures, allLabels, outputFileLocation, persistenceType)
    print "Persistence complete.  Generated featured data for labeled images."
    
    

def generateLabeledImageFeatures(msrcImage, ignoreVoid=False):
    """This function takes an msrcImage object and returns a 2-element list.  The first element contains an array of pixel feature values, the second contains an array of pixel class label.
    The ignoreVoid flag is used to handle the void class label; when True void class pixels are not included in result set, when False void pixels are included."""
    # Make me user input!
    numGradientBins = 9
    numHistBins = 12
    
    if ignoreVoid == False:
        # Just process all pixels, whether void or not
        allPixelFeatures = generatePixelFeaturesForImage(msrcImage.m_img, numGradientBins, numHistBins)
        allPixelLabels = reshapeImageLabelData(msrcImage)
        
        assert (np.size(allPixelLabels) == np.shape(allPixelFeatures)[0] ), ("Image pixel labels & features are different size! labelSize="\
                                                                            + str(np.size(allPixelLabels)) + ", featureSize=" + str(np.size(allPixelFeatures[0])) + "")
        return [ allPixelFeatures, allPixelLabels ]
        
    else:
        # Need to check result pixel before inclusion in result feature vector
        voidIdx = pomio.msrc_classLabels.index("void")
        nonVoidFeatures = None
        nonVoidLabels = None
        
        allPixelFeatures = generatePixelFeaturesForImage(msrcImage.m_img, numGradientBins, numHistBins)
        allPixelLabels = reshapeImageLabelData(msrcImage)
        
        assert (np.size(allPixelLabels) == np.shape(allPixelFeatures)[0] ), ("Image pixel labels & features are different size! labelSize="\
                                                                            + str(np.size(allPixelLabels)) + ", featureSize=" + str(np.size(allPixelFeatures[0])) + "")
        
        # check each pixel label, add to result list iff != void
        numFeatures = np.shape(allPixelFeatures)[1]
        
        print "number of features = " , numFeatures
        
        # get boolean array of labels != void index
        nonVoidLabelCondition = (allPixelLabels != voidIdx)
        print "Shape of boolean array of non-void labels = " , np.shape(nonVoidLabelCondition)
        # can use 1d boolean as index to select non-void labels
        nonVoidLabels = allPixelLabels[nonVoidLabelCondition]
        print "Shape of non void labels = " , np.shape(nonVoidLabels)
        print "Now getting row indices of non-zero labels..."
        nonVoidRowIdxs = np.arange(0, np.size(allPixelLabels))[nonVoidLabelCondition]
        
        print "Now getting non-void features from row indices::"
        nonVoidFeatures = allPixelFeatures[nonVoidRowIdxs]
        print "Non-void feature array shape = " , np.shape(nonVoidFeatures)
        print "Non-void labels array shape = " , np.shape(nonVoidLabels)
        
        assert (np.size(nonVoidLabels) == np.shape(nonVoidFeatures)[0] ), ("Non-void pixel label & feature data are different size! Non void labelSize=" + str(np.size(nonVoidLabels)) + ", Non void featureSize=" + str(np.shape(nonVoidFeatures)[0]) + "")
        print "Extracted non void features and labels, assigned to result variable"
        return [nonVoidFeatures, nonVoidLabels]


def generatePixelFeaturesForImage(rgbSourceImage, numGradientBins, numHistBins):
    """This function takes an RGB image as numpy (i,j, 3) array as input and returns pixel-wise features (i * j , numFeatures) array.
    numGraidentBins is used in Historgram of Orientation (HOG) feature generation.
    numHistBins is used in the colour histogram feature generation (RGB & HSV)"""
    totalImagePixels = np.size(rgbSourceImage[:,:,0])
    
    # RGB features
    allRed = np.reshape(rgbSourceImage[:,:,0] , (totalImagePixels, 1) )
    allGreen = np.reshape(rgbSourceImage[:,:,1] , (totalImagePixels, 1) )
    allBlue = np.reshape(rgbSourceImage[:,:,2] , (totalImagePixels, 1) ) 
    rgbColourValuesFeature = np.hstack( ( allRed, allGreen, allBlue ) )
        
    rgbColour3DHistogramFeatures = FeatureGenerator.create3dRGBColourHistogramFeature(rgbSourceImage, numHistBins)
    rgbColour3DHistogramFeatures = np.resize(rgbColour3DHistogramFeatures, (totalImagePixels, np.size(rgbColour3DHistogramFeatures[1]) ) )
        
    rgbColour1DHistogramFeatures = FeatureGenerator.create1dRGBColourHistogram(rgbSourceImage, numHistBins)
    rgbColour1DHistogramFeatures = np.resize(rgbColour1DHistogramFeatures, (totalImagePixels, np.size(rgbColour1DHistogramFeatures[1]) ) )
        
        
    # HSV features
    hsvSourceImage = color.rgb2hsv(rgbSourceImage)
    allHuePixels = np.reshape(hsvSourceImage[:,:,0] , (totalImagePixels, 1) )
    allSaturationPixels = np.reshape(hsvSourceImage[:,:,1] , (totalImagePixels, 1) )
    allValueBrightPixels = np.reshape(hsvSourceImage[:,:,2] , (totalImagePixels, 1) ) 
    hsvColourValueFeatures = np.hstack( ( allHuePixels, allSaturationPixels, allValueBrightPixels ) )
        
    hsvColour1DHistogramFeatures = FeatureGenerator.create1dHSVColourHistogram(hsvSourceImage, numHistBins) 
    hsvColour1DHistogramFeatures = np.resize(hsvColour1DHistogramFeatures, (totalImagePixels, np.size(hsvColour1DHistogramFeatures[1]) ) )
        
    hsvColour3DHistogramFeatures = FeatureGenerator.create3dHSVColourHistogramFeature(hsvSourceImage, numHistBins)
    hsvColour3DHistogramFeatures = np.resize(hsvColour3DHistogramFeatures, (totalImagePixels, np.size(hsvColour3DHistogramFeatures[1]) ) )
     
    # HOG features
    hog1Darray, hogFeatures = FeatureGenerator.createHistogramOfOrientedGradientFeatures(rgbSourceImage, numGradientBins, (8,8), (3,3), True, True)
    hog1Darray = None
    hogFeatures = np.reshape(hogFeatures, (totalImagePixels , np.size(hogFeatures) / totalImagePixels) )
    
    # Local binary pattern features
    lbpFeatures = FeatureGenerator.createLocalBinaryPatternFeatures(rgbSourceImage, 4, 2, "default")
    lbpFeatures = np.reshape(lbpFeatures, (totalImagePixels , np.size(lbpFeatures) / totalImagePixels) )
   
    # Testure filter response features
    filterResponseFeatures = FeatureGenerator.createFilterbankResponse(rgbSourceImage, 15)
    filterResponseFeatures = np.reshape(filterResponseFeatures, ( totalImagePixels , np.size(filterResponseFeatures) / totalImagePixels))
    
    # Consolidate all features for image, per pixel
    imageFeatures = np.hstack( [rgbColourValuesFeature, rgbColour1DHistogramFeatures, rgbColour3DHistogramFeatures, \
                                    hsvColourValueFeatures, hsvColour1DHistogramFeatures , hsvColour3DHistogramFeatures,  \
                                    hogFeatures, \
                                    lbpFeatures, \
                                    filterResponseFeatures ] )
    
    return imageFeatures


#
# Prediction util methods
#

def generatePredictedPixelClassDist(rgbImage, classifier, numberLabels):
    """This image taks an RGB image as an (i,j,3) numpy array, a scikit-learn classifier and produces probability distribution over each pixel and class.
    Returns an (i,j,N) numpy array where N= total number of classes for use in subsequent modelling."""
    
    # Take image, generate features, use classifier to predict labels, ensure normalised dist and shape to (i,j,N) np.array
    numGradientBins = 9
    numHistBins = 16
    
    imagePixelFeatures = generatePixelFeaturesForImage(rgbImage, numGradientBins, numHistBins)
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
    writeAllFeatureDataToCsv(features, str(baseFileLocation + "Data.csv"))
    writeAllLabelDataToCsv(labels , str(baseFileLocation + "Labels.csv"))
 
def writeAllFeatureDataToCsv(features, csvFilename):
    # accept a 1d array
    for idx in range(0 , np.size(features[0])):
        writeArrayDataToCsvRow(features[idx], csvFilename)

def writeAllLabelDataToCsv(labels, csvFilename):
    for idx in range(0 , np.size(labels[0])):
        writeArrayDataToCsvRow(labels[idx], csvFilename)

def writeArrayDataToCsvRow(features, csvFilename):
    # http://stackoverflow.com/questions/12218945/formatting-numpy-array
    # read back with np.fromstring(s.getvalue(), sep=',')
    sio = StringIO()
    dataFile = open(csvFilename, 'a')
    
    print "Writing features to file:: " + str(np.shape(features))
    np.savetxt(sio, features, fmt='%.10f', delimiter=',')
    
    dataFile.write(sio.getvalue())
    dataFile.flush()
    
    dataFile.close()

def readLabeledDataFromCsv(baseFilename):
    featureData = readArrayDataFromCsvFile(baseFilename + "Data.csv")
    labelData = readArrayDataFromCsvFile(baseFilename + "Labels.csv")
    return [featureData, labelData]

def readArrayDataFromCsvFile(arrayDataFile):
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

# Pickle serialisation utils

def pickleLabeledDataToFile(features, labels, baseFileLocation):
    pickleNumpyData(features, str(baseFileLocation + "Data.npy"))
    pickleNumpyData(labels , str(baseFileLocation + "Labels.npy"))
    
def pickleNumpyData(data, filename):
    print "Pickle my shizzle"
    np.save(filename, data)
    print "Shizzle is now pickled"

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
trainingPixelBaseFilename = "/home/amb/dev/mrf/classifier/logisticRegression/pixelLevelModel/training/trainPixelFeature"
validationPixelBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModel/crossValidation/data/validPixelFeature"
validationResultsBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/crossValidation/results/logRegClassifier_CrossValidResult"

testingPixelBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModel/testing/data/testPixelFeature"
testingResultsBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModel/testing/results/testPixelFeature"

classifierBaseFilename = "/home/amb/dev/mrf/classifiers/logisticRegression/pixelLevelModels/classifierModels/logRegClassifier"

# Generate data from images and save to file
splitData = splitInputDataset_msrcData(msrcData, keepClassDist=False, train=0.6, validation=0.2, test=0.2)
trainData = splitData[0]
validationData = splitData[1]
testData = splitData[2]

# Create & persist feature & label data for training, cross-validation of C apram and classifier testing
processLabeledImageData(trainData, trainingPixelBaseFilename, persistenceType="csv" , ignoreVoid=True)
processLabeledImageData(validationData, validationPixelBaseFilename, persistenceType="csv", ignoreVoid=True)
processLabeledImageData(testData, testingPixelBaseFilename, persistenceType="csv", ignoreVoid=True)

# Load generated data for classification
trainingData = readLabeledDataFromCsv(trainingPixelBaseFilename)
validationData = readLabeledDataFromCsv(validationPixelBaseFilename)
testingData = readLabeledDataFromCsv(testingPixelBaseFilename)

# cross-validation on C param
C_min = 0.1
C_max = 1
C_increment = 0.1
crossValidation_Cparam(trainingData, validationData, classifierBaseFilename, validationResultsBaseFilename, C_min, C_max, C_increment)

