import os
import sys
import datetime
import numpy as np
from cStringIO import StringIO
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pickle
import pomio, FeatureGenerator
import matplotlib.pyplot as plt
import matplotlib
import NeuralNet
import sklearn.ensemble

def trainLogisticRegressionModel(
    featureData, labels, Cvalue, outputClassifierFile, scaleData=True, requireAllClasses=True
    ):
    # See [http://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html]
    # Features are numPixel x unmFeature np arrays, labels are numPixel np array
    numTrainDataPoints = np.shape(featureData)[0]
    numDataLabels = np.size(labels)
    
    assert ( np.size( np.shape(labels) ) == 1) , ("Labels should be a 1d array.  Shape of labels = " + str(np.shape(labels)))
    assert ( numTrainDataPoints == numDataLabels) , ("The length of the feature and label data arrays must be equal.  Num data points=" + str(numTrainDataPoints) + ", labels=" + str(numDataLabels) )
    classLabels = np.unique(labels)
    assert not requireAllClasses or \
        ( np.size(classLabels) == 23 or np.size(classLabels) == 24 ), \
        "Training data does not contains all classes::\n\t" + str(classLabels)
     
    if scaleData == True:
        featureData = preprocessing.scale(featureData)
    
    # sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    lrc = LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=Cvalue, fit_intercept=True, intercept_scaling=1)
    lrc.fit(featureData, labels)
    
    pickleObject(lrc, outputClassifierFile)
    print "LogisticRegression classifier saved to " + str(outputClassifierFile)
    
    return lrc


def testClassifier(classifier, testFeatureData, testLabels, resultsFile, scaleData=True):
    # predict on testFeatures, compare to testClassLabels, return the results
    # TODO implement "ignore void class"
    accuracy = classifier.score(testFeatureData, testLabels)
    return accuracy


def crossValidation_Cparam(trainingData, validationData, classifierBaseFile, classifierBaseTestOutputFile, C_min, C_max, C_increment):
    
    assert (C_min <= C_max) , "C_min param value should be less than or equal to C_max param value."
    assert C_increment != 0, "You specified min and max for C param; C_increment value must NOT be 0"
    assert (np.size(np.shape(trainingData)) == 2 and np.size(np.shape(validationData)) == 2) , ("Training data and test data must be list of length 2 (first element giving feature array, second giving label array. #Training=" + str(np.size(np.shape(trainingData)) + ", #test=" + str(np.shape(validationData)))) 
    
    # Get training data
    trainFeatureData = trainingData[0]
    trainLabels = trainingData[1]
    numTrainData = np.shape(trainFeatureData)[0]
    numTrainLabels = np.size(trainLabels)
    
    assert ( numTrainData == numTrainLabels ) , "TRAIN data error:: Number of feature vectors must equal number of labels. #trainingFeatures=" + ""
    
    # Get test data
    validationFeatureData = validationData[0]
    validationLabels = validationData[1]
    numValidData = np.shape(validationFeatureData)[0]
    numValidLabels = np.size(validationLabels)
    
    assert ( numValidData == numValidLabels ) , "VALIDATION data error:: Number of feature vectors must equal number of labels. #trainingFeatures=" + ""
    
    print "Now training classifiers for specified C params::"
    
    cvResult = None
    
    # Just train and test on single C_value
    if C_min == C_max:
        
        print "\nTraining classifier C="+str(C_min)
        classifierName = classifierBaseFile + "_" + str(C_min) + ".pkl"
        if C_min <= 0:
                print "C-param is 0 - must be greater than 0.  Added 10^-6 to give 0.000001 as first param value"
                C_min = 0.0 + 10**(-6)
                
        classifier = trainLogisticRegressionModel(trainFeatureData, trainLabels, 0.1, classifierName, True)
        meanAccuracy = testClassifier(classifier, validationFeatureData, validationLabels, classifierBaseTestOutputFile + "_" + str(C_min) + ".csv", True)
        
        cvResult = np.array( [ C_min, meanAccuracy])
        return cvResult
    
    else:

        # Now run cross-validation on each C param, persisting classifier CV performance and corresponding classifier object to file
        Crange = np.arange(0, C_max+C_increment, C_increment)
        print "Training classifiers with C params in range:" , Crange
         
        for idx in range(0, len(Crange)):
            C_param = Crange[idx]
            if C_param <= 0:
                print "Min C_param is 0 - must be greater than 0.  Added 10^-6 to give 0.000001 as first param value"
                C_param = 0.0 + 10**(-6)
            
            print "\nTraining classifier C="+str(C_param)
            classifier = trainLogisticRegressionModel(trainFeatureData, trainLabels, C_param, classifierBaseFile + "_" + str(C_param) + ".pkl", True)
            trainAccuracy = classifier.score(trainFeatureData, trainLabels)
            print "Classifier trained, now using scikit learn test function, trainingAccuracy = " , (trainAccuracy * 100) , "%"
            cvAccuracy = testClassifier(classifier, validationFeatureData, validationLabels, classifierBaseTestOutputFile + str(C_param) + "_" + ".csv", True)
            
            print "LR classifier, C_para =" , C_param , " cv_accuracy = ", (cvAccuracy * 100) , "%"
            if cvResult == None:
                cvResult = np.array( [C_param , trainAccuracy, cvAccuracy ])
            else:
                cvResult = np.vstack( [ cvResult , np.array( [ C_param, trainAccuracy, cvAccuracy ] ) ] )
        return cvResult


#
# Prediction util methods
#

def generateImagePredictionClassDist(rgbImage, classifier, requireAllClasses=True):
    """This image takes an RGB image as an (i,j,3) numpy array, a scikit-learn classifier and produces probability distribution over each pixel and class.
    Returns an (i,j,N) numpy array where N= total number of classes for use in subsequent modelling."""
    
    # TODO Broaden to cope with more classifiers :)
    #assert (str(type(classifier)) == "<class 'sklearn.linear_model.logistic.LogisticRegression'>") , "Check classifier type value:: " + str(type(classifier)) 
    testClassifier = None
    
    imageDimensions = rgbImage[:,:,0].shape
    nbCols = imageDimensions[1]
    nbRows = imageDimensions[0]
    params = classifier.get_params(deep=True)
    
    print "Classifier paras::" , params
    
    # Take image, generate features, use classifier to predict labels, ensure normalised dist and shape to (i,j,N) np.array
    
    # generate predictions for the image
    imagePixelFeatures = FeatureGenerator.generatePixelFeaturesForImage(rgbImage)
    #print imagePixelFeatures
    predictedPixelLabels = classifier.predict(imagePixelFeatures)
    predictionProbs = classifier.predict_proba(imagePixelFeatures)
    print "\nShape of predicted labels::" , np.shape(predictedPixelLabels)
    print "\nShape of prediction probs::" , np.shape(predictionProbs)
    numClasses = pomio.getNumClasses()
    
    assert not requireAllClasses or \
        (np.shape(predictionProbs)[1] == numClasses or \
             np.shape(predictionProbs)[1] == numClasses-1) , \
             "Classifer prediction does not match all classes (23 or 24):: " + \
             str(np.shape(predictionProbs)[1])
    print predictionProbs
    
    #!!predictionProbs = np.reshape(predictionProbs, (nbCols, nbRows, numClasses ))
    print 'reshaping to ', (nbCols, nbRows, predictionProbs.shape[1] )
    predictionProbs = np.reshape(predictionProbs, (nbRows, nbCols, predictionProbs.shape[1] ))
    
    return predictionProbs



#
# File IO util functions
#

def persistLabeledDataToFile(features, labels, baseFileLocation, persistenceType):
    
    assert (persistenceType == "numpy" or persistenceType == "pickle" or persistenceType == "csv"), "persistenceType must be string value \"pickle\" or \"csv\""
    
    if persistenceType == "csv":
        writeLabeledDataToCSVFile(features, labels, baseFileLocation)
    
    elif persistenceType == "numpy":
        saveLabeledDataToNumpyFile(features, labels, baseFileLocation)
    
    elif persistenceType == "pickle":
        info = [ features, labels ]
        pickleObject(info , baseFileLocation)

#
# CSV file utils
#

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


# Numpy serialisation utils

def saveLabeledDataToNumpyFile(features, labels, baseFileLocation):
    saveNumpyData(features, str(baseFileLocation + "Data.npy"))
    saveNumpyData(labels , str(baseFileLocation + "Labels.npy"))
    
def saveNumpyData(data, filename):
    np.save(filename, data)

def readLabeledNumpyData(baseFileLocation):
    features = loadNumpyData(str(baseFileLocation + "Data.npy"))
    labels = loadNumpyData(str(baseFileLocation + "Labels.npy"))
    return [features, labels]
    
def loadNumpyData(filename):
    data = np.load( open(filename , "r") )
    return data


# Classifier IO utils
def pickleObject(obj, filename):
    if filename.endswith(".pkl"):
        f = open( filename , "w")
        pickle.dump(obj, f , True)
        f.close()
    else:
        print "Input filename did not end in .pkl - adding .pkl to filename."
        filename= str(filename)+".pkl"
        f = open( filename , "w")
        pickle.dump(obj, f , True)
        f.close()
    
    
def loadObject(filename):
    filetype = '.pkl'
    if filename.endswith(filetype):
        f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    else:
        print "Input filename did not end in .pkl - trying filename with type appended...."
        f = open( ( str(filename)+".pkl" ), "rb")
        obj = pickle.load(f)
        f.close()
        return obj

def loadClassifier(fullFilename):
    return loadObject(fullFilename)

def scaleInputData(inputFeatureData):
    # Assumes numeric numpy array [[ data....] , [data....] ... ]
    return preprocessing.scale(inputFeatureData.astype('float'))

def makedear( base, dirname ):
    fn = base + dirname
    if not os.path.isdir( fn ):
        os.mkdir(fn)

if __name__ == "__main__":

    #
    # Simple runtime tests
    #
    # TODO look at sklearn pipeline to get some automation here
    doVal = 0
    doTest = 0
    
    msrcData = sys.argv[1] #"/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"
    timestamp = str(datetime.datetime.now())
    # Output file resources
    outDir = sys.argv[2]
    trainingPixelBaseFilename     = outDir + "/training/trainPixelFeature" 
    validationPixelBaseFilename   = outDir + "/crossValidation/data/cvPixelFeature"
    validationResultsBaseFilename = outDir + "/crossValidation/results/logRegClassifier_CrossValidResult"
     
    testingPixelBaseFilename      = outDir + "/testing/data/testPixelFeature"
    testingResultsBaseFilename    = outDir + "/testing/results/testPixelFeature"
     
    #classifierBaseFilename        = outDir + "/classifierModels/logRegClassifier"
    classifierBaseFilename        = outDir + "/classifierModels/randForestClassifier"
    
    makedear( outDir, "" )
    makedear( outDir, "/training" )
    makedear( outDir, "/crossValidation" )
    makedear( outDir, "/crossValidation/data" )
    makedear( outDir, "/crossValidation/results" )
    makedear( outDir, "/testing" )
    makedear( outDir, "/testing/data" )
    makedear( outDir, "/testing/results" )
    makedear( outDir, "/classifierModels" )

    # Load dem images
    if 0:
        msrcImages = pomio.msrc_loadImages(msrcData, [\
                'Images/10_1_s.bmp',\
                'Images/10_2_s.bmp',\
                'Images/11_1_s.bmp',\
                'Images/11_2_s.bmp',\
                'Images/12_1_s.bmp',\
                'Images/12_2_s.bmp',\
                'Images/13_1_s.bmp',\
                'Images/13_2_s.bmp',\
                'Images/14_1_s.bmp',\
                'Images/14_2_s.bmp',\
                'Images/15_1_s.bmp',\
                'Images/15_2_s.bmp',\
                'Images/16_1_s.bmp',\
                'Images/16_2_s.bmp',\
                'Images/17_1_s.bmp',\
                'Images/17_2_s.bmp',\
                'Images/18_1_s.bmp',\
                'Images/18_2_s.bmp',\
                'Images/19_1_s.bmp',\
                'Images/19_2_s.bmp',\
                'Images/1_1_s.bmp',\
                'Images/1_2_s.bmp',\
                'Images/20_1_s.bmp',\
                'Images/20_2_s.bmp',\
                'Images/2_1_s.bmp',\
                'Images/2_2_s.bmp',\
                'Images/3_1_s.bmp',\
                'Images/3_2_s.bmp',\
                'Images/4_1_s.bmp',\
                'Images/4_2_s.bmp',\
                'Images/5_1_s.bmp',\
                'Images/5_2_s.bmp',\
                'Images/6_1_s.bmp',\
                'Images/6_2_s.bmp',\
                'Images/7_1_s.bmp',\
                'Images/7_2_s.bmp',\
                'Images/8_1_s.bmp',\
                'Images/8_2_s.bmp',\
                'Images/9_1_s.bmp',\
                'Images/9_2_s.bmp'])
    else:
        msrcImages = pomio.msrc_loadImages(msrcData, ['Images/7_3_s.bmp'] )

    if doVal or doTest:
        scale = 0.1
        # Generate data from images and save to file
        print "\nProcessing " + str(scale*100) + \
            "% of MSRC data on a 60/20/20 split serialised for easier file IO"
        splitData = pomio.splitInputDataset_msrcData(
            msrcImages,
            datasetScale=scale,
            keepClassDistForTraining=True,
            trainSplit=0.6,
            validationSplit=0.2,
            testSplit=0.2
            )
        
        validationDataset = splitData[1]
        testDataset = splitData[2]
        
        
        if doVal:
            print "Processing validation data::"
            validationData = FeatureGenerator.processLabeledImageData(validationDataset, ignoreVoid=True)
        
        if doTest:
            print "Processing test data::"
            testingData = FeatureGenerator.processLabeledImageData(testDataset, ignoreVoid=True)
    else:
        # Just training data
        splitData = [msrcImages,None,None]

    # prepare training data
    trainDataset = splitData[0]
    
    alTrainlLabels = None
    
    for idx in range(0, np.size(trainDataset)):
        if alTrainlLabels == None:
            alTrainlLabels = FeatureGenerator.reshapeImageLabels(trainDataset[idx])
        else:
            alTrainlLabels = np.append( alTrainlLabels , FeatureGenerator.reshapeImageLabels(trainDataset[idx]) )
    print "\nProcessing training data::"
    trainingData = FeatureGenerator.processLabeledImageData(\
        trainDataset, ignoreVoid = True, nbPerImage = 1000)
        

    # # try to save the data
    # print "\nTry to pickle the results..."
    # pickleObject(trainingData, trainingPixelBaseFilename) # Why you fail me Mr Pickle?
    # pickleObject(validationData, validationPixelBaseFilename)
    # pickleObject(testingData, testingPixelBaseFilename)
    
    if doVal:
        # cross-validation on C param
        print "\nNow using validation data set to evaluate different C param values @" , datetime.datetime.now()
        C_min       = 0
        C_max       = 1.0
        C_increment = 0.5
        cvResult    = crossValidation_Cparam(trainingData, validationData, classifierBaseFilename, validationResultsBaseFilename, C_min, C_max, C_increment)
        print "Completed @ " + str(datetime.datetime.now()), "\nCV results for different C params:\n" , cvResult
    else:
        # Use fixed C
        C           = 0.5
        # Just train on a subset!!
        nbToTrainOn = 1000
        print 'TRAINING CLASSIFIER on %d-sample subset of %d examples of dimension %d...' % \
            ( nbToTrainOn, trainingData[0].shape[0], trainingData[0].shape[1] )
        subset = np.random.choice( trainingData[0].shape[0], nbToTrainOn, replace=False )
        if 0:
            # logistic regression
            classifier = trainLogisticRegressionModel(
                trainingData[0][subset,:], trainingData[1][subset], C, classifierBaseFilename, \
                    scaleData=True, \
                    requireAllClasses=False
                )
        elif 0:
            # Neural Network
            # Construct nn dataset
            datmat = trainingData[0][subset,:]
            labvec = trainingData[1][subset]
            nbFeatures = datmat.shape[1]
            nbClasses = 23#np.max(labvec)+1
            nbHidden = 100
            maxIter = 200
            classifier = NeuralNet.NNet(nbFeatures, nbClasses, nbHidden)
            nnds = classifier.createTrainingSetFromMatrix( datmat, labvec )
            classifier.trainNetworkBackprop(nnds,maxIter)
        else:
            #classifier = None
            # Random forest
            datmat = trainingData[0][subset,:]
            labvec = trainingData[1][subset]
            print '**Training a random forest on %d examples...' % len(labvec)
            print 'Labels represented: ', np.unique( labvec )
            classifier = sklearn.ensemble.RandomForestClassifier(\
                n_estimators=100)
            classifier = classifier.fit( datmat, labvec )
            pickleObject(classifier, classifierBaseFilename)
            print "Rand forest classifier saved to " + str(classifierBaseFilename)

    # 
    #     classifierVersion = "_0.5"
    #     filetype = ".pkl"
    #     classifierFilename = classifierBaseFilename + classifierVersion + filetype
    #     
    #     print "Reading pre-built classifier from file::" , classifierFilename , "@" , datetime.datetime.now()
    #     classifier = loadClassifier(classifierFilename)
    #     
    
    # predictImage = pomio.msrc_loadImages(msrcData)[0]
    plt.interactive(1)
    plt.figure()
    pomio.showClassColours()
    plt.figure()
    for img in msrcImages:
        print "\nRead in image %s from the MSRC dataset::" % img.m_imgFn
        imageFeatures = FeatureGenerator.generatePixelFeaturesForImage(img.m_img)
        predLabs = classifier.predict(imageFeatures)
        print "\nGenerating prediction of shape ", predLabs.shape, "::" , predLabs
        predImg = np.reshape( predLabs, img.m_img[:,:,0].shape )
        #print "\nGenerating the probability dist. for each pixel over class labels @" , datetime.datetime.now()
        #imageClassDist = generateImagePredictionClassDist(img.m_img, classifier,False)
        
        # Display
        print "Unique labels from clfr = ", np.unique(predLabs)
        plt.subplot(1,2,1)
        plt.imshow(img.m_img)
        plt.subplot(1,2,2)
        pomio.showLabels( predImg )
        plt.waitforbuttonpress()

    print "\tCompleted @ " + str(datetime.datetime.now())

plt.interactive(0)
plt.show()
