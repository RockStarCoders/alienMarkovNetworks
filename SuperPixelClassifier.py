import numpy as np

from scipy import stats

import pickle

import sklearn.ensemble
from sklearn.linear_model import LogisticRegression

import skimage
import skimage.data

import matplotlib.pyplot as plt
import matplotlib

import pomio
import SuperPixels
import FeatureGenerator

import pdb

import bonzaClass

# Create a random forest classifier for superpixel class given observed image features
# 1) Get labelled training data
# 2) Create single training dataset for each superpixel in each training image:    data = [superPixelFeatures , superPixelLabels]
# 3) Create random forest classifier from data
# 4) Save classifier to file



# Use this if you have generated a set of MSRC image to process
def getSuperPixelData(msrcImages,numberSuperPixels, superPixelCompactness):
    
    # Should probably make this a call to pomio in case the ordering changes in the future...
    voidClassLabel = pomio.getVoidIdx()
    
    numberImages = len(msrcImages)    
    
    # for each image:
    #   determine superpixel label (discard if void)
    #   compute superpixel features of valid superpixels
    #   append features to cumulative array of all super pixel features
    #   append label to array of all labels
    
    superPixelFeatures = None
    superPixelLabels = np.array([], int) # used for superpixel labels
    numberVoidSuperPixels = 0   # keep track of void superpixels

    nbClasses = pomio.getNumClasses()
    classAdjCounts = np.zeros( (nbClasses, nbClasses) )
    adjCountsTotal = 0
    adjVoidCountsTotal = 0

    for imgIdx in range(0, numberImages):
    
        superPixelIgnoreList = np.array([], int) # this is used to skip over the superpixel in feature processing
    
        print "\n**Processing Image#" , (imgIdx + 1) , " of" , numberImages
    
        # get raw image and ground truth labels
        img = msrcImages[imgIdx].m_img
        imgPixelLabels = msrcImages[imgIdx].m_gt
        
        # create superpixel map and graph for image
        spgraph = SuperPixels.computeSuperPixelGraph( img, 'slic', [numberSuperPixels, superPixelCompactness] )
        imgSuperPixelMask = spgraph.m_labels
        imgSuperPixels = spgraph.m_nodes
        numberImgSuperPixels = spgraph.getNumSuperPixels()
    
        # create superpixel exclude list & superpixel label array
        allSPClassLabels = []
        for spIdx in range(0, numberImgSuperPixels):
            
            superPixelValue = imgSuperPixels[spIdx]
            #print "\tINFO: Processing superpixel =", superPixelValue , " of" , numberImgSuperPixels, " in image"
            
            
            # Assume superpixel labels are sequence of integers
            superPixelValueMask = (imgSuperPixelMask == superPixelValue ) # Boolean array for indexing superpixel-pixels
            superPixelLabel = assignClassLabelToSuperPixel(superPixelValueMask, imgPixelLabels)
            allSPClassLabels.append( superPixelLabel)

            if(superPixelLabel == voidClassLabel):
            
                # add to ignore list, increment void count & do not add to superpixel label array
                superPixelIgnoreList = np.append(superPixelIgnoreList, superPixelValue)
                numberVoidSuperPixels = numberVoidSuperPixels + 1
                
            else:
                superPixelLabels = np.append(superPixelLabels, superPixelLabel)
        
        assert len(allSPClassLabels) == numberImgSuperPixels
        (theseClassAdjCounts,adjVoidCount,adjCount) = spgraph.countClassAdjacencies( nbClasses, allSPClassLabels )
        classAdjCounts     += theseClassAdjCounts
        adjCountsTotal     += adjCount
        adjVoidCountsTotal += adjVoidCount

        # Now we have the superpixel labels, and an ignore list of void superpixels - time to get the features!
        imgSuperPixelFeatures = FeatureGenerator.generateSuperPixelFeatures(img, imgSuperPixelMask, excludeSuperPixelList=superPixelIgnoreList)
        
        if superPixelFeatures == None:        
            superPixelFeatures = imgSuperPixelFeatures;
        else:
            # stack the superpixel features into a single list
            superPixelFeatures = np.vstack( [ superPixelFeatures, imgSuperPixelFeatures ] )
    
    
    assert np.shape(superPixelFeatures)[0] == np.shape(superPixelFeatures)[0] , "Number of samples != number labels"
    print "\n**Processed total of" , numberImages, "images"
    print "  %d out of %d adjacencies were ignored due to void (%.2f %%)" % \
        (adjVoidCountsTotal, adjCountsTotal, \
             100.0*adjVoidCountsTotal/adjCountsTotal)

    # Now return the results
    return [ superPixelFeatures, superPixelLabels, classAdjCounts ]



# use this function to generate features and labels for default training dataset split
def getSuperPixelTrainingData(msrcDataDirectory, nbSuperPixels,superPixelCompactness, scale, trainSplit=0.6, validationSplit=0.2, testSplit=0.2):
    
    # Should probably make this a call to pomio in case the ordering changes in the future...
    voidClassLabel = pomio.getVoidIdx()
    
    # These could be user-specified
    if scale == None:
        scale = 0.05 # default to 10% of data
        
    msrcData = pomio.msrc_loadImages(msrcDataDirectory, None)

    print "Now generating superpixel classifier for MSRC data"
    
#    splitData = pomio.splitInputDataset_msrcData(msrcData, datasetScale=scale, keepClassDistForTraining=True, trainSplit, validationSplit, testSplit )
    splitData = pomio.splitInputDataset_msrcData(msrcData, scale, True, trainSplit, validationSplit, testSplit )
    
    # prepare superpixel training data
    trainingMsrcImages = splitData[0]
    
    # Just use the above function to get superpixel features and labels for training data
    return SuperPixelClassifier.getSuperPixelData(trainingMsrcImages,nbSuperPixels,superPixelCompactness)
    
    


# Could use other params e.g. training dataset size
def createSuperPixelRandomForestClassifier(msrcDataDirectory, classifierDirectory,nbSuperPixels,superPixelCompactness, scale):
    
    # Get training data
    superPixelTrainData = getSuperPixelTrainingData(msrcDataDirectory,nbSuperPixels,superPixelCompactness, scale)
    superPixelTrainFeatures = superPixelTrainData[0]
    superPixelTrainLabels = superPixelTrainData[1]
    
    # now train random forest classifier on labelled superpixel data
    print '\n\n**Training a random forest on %d examples...' % len(superPixelTrainLabels)
    print "Training feature data shape=" , np.shape(superPixelTrainFeatures) , "type:", type(superPixelTrainFeatures)
    print "Training labels shape=" , np.shape(superPixelTrainLabels), "type:" , type(superPixelTrainLabels)

    numEstimators = 100
    
    classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=numEstimators)
    classifier = classifier.fit( superPixelTrainData, superPixelTrainLabels)
    
    # Now serialise the classifier to file
    classifierFilename = classifierDirectory + "/" + "randForest_superPixel_nEst" + str(numEstimators) + ".pkl"
    
    pomio.pickleObject(classifier, classifierFilename)
    
    print "Rand forest classifier saved @ " + str(classifierFilename)



def createSuperPixelLogisticRegressionClassifier(msrcDataDirectory, classifierDirectory, scale):
    
    # Get training data
    superPixelTrainData = getSuperPixelTrainingData(msrcDataDirectory,nbSuperPixels,superPixelCompactness, scale)
    superPixelTrainFeatures = superPixelTrainData[0]
    superPixelTrainLabels = superPixelTrainData[1]
    
    # now train random forest classifier on labelled superpixel data
    print '\n\n**Training a LR classifier on %d examples...' % len(superPixelTrainLabels)
    print "Training feature data shape=" , np.shape(superPixelTrainFeatures) , "type:", type(superPixelTrainFeatures)
    print "Training labels shape=" , np.shape(superPixelTrainLabels), "type:" , type(superPixelTrainLabels)

    Cvalue = 0.5
    # sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    classifier = LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=Cvalue, fit_intercept=True, intercept_scaling=1)
    classifier.fit(superPixelTrainFeatures, superPixelTrainLabels)
    
    
    # Now serialise the classifier to file
    classifierFilename = classifierDirectory + "/" + "LR_superPixel_C" + str(Cvalue) + ".pkl"
    
    pomio.pickleObject(classifier, classifierFilename)
    
    print "LR classifier saved @ " + str(classifierFilename)



# If makeProbabilities is true, the third output arg is the prob
# matrix. Otherwise, None.
def predictSuperPixelLabels(classifier, image,numberSuperPixels,  \
                                superPixelCompactness, makeProbabilities ):
    print "\n**Computing super pixel labelling for input image"
    outProbs = None

    # Get superpixels
    spgraph = SuperPixels.computeSuperPixelGraph(image,'slic',[numberSuperPixels, superPixelCompactness])
    imgSuperPixelsMask = spgraph.m_labels
    imgSuperPixels = spgraph.m_nodes
    numberImgSuperPixels = len(imgSuperPixels)
    print "**Image contains", numberImgSuperPixels, "superpixels"
    
    # Get superpixel features
    superPixelFeatures = FeatureGenerator.generateSuperPixelFeatures(image, imgSuperPixelsMask, None)
    assert np.shape(superPixelFeatures)[0] == numberImgSuperPixels, "Number of superpixels in feature array != number super pixels in image!:: " + str(np.shape(superPixelFeatures)[0]) + " vs. " + str(numberImgSuperPixels)

    superPixelLabels = classifier.predict( superPixelFeatures )
    
    if makeProbabilities:
        outProbs = bonzaClass.classProbsOfFeatures(
            superPixelFeatures, classifier, requireAllClasses=False
        )
    return (superPixelLabels, spgraph, outProbs)



def getSuperPixelLabelledImage(image, superPixelMask, superPixelLabels):
    superPixels = np.unique(superPixelMask)
    numSuperPixels = np.shape(superPixels)[0]
    numSuperPixelLabels = np.shape(superPixelLabels)[0]
    assert numSuperPixels == numSuperPixelLabels , "The number of superpixels in mask != number superpixel labels:: " + str(numSuperPixels) + " vs. " + str(numSuperPixelLabels) 
    
    labelImage = np.zeros( np.shape(image[:,:,0]) )
    
    for idx in range(0, numSuperPixels):
        # Assume a consistent ordering between unique superpixel values and superpixel labels
        labelImage = labelImage + ( superPixelLabels[idx] * (superPixelMask==superPixels[idx]).astype(int) )
    
    assert np.unique(labelImage).all() == np.unique(superPixelLabels).all() , "List of unique class labels in image != image labels:: " + str(np.unique(labelImage)) + " vs. " + str(superPixelLabels)
    
    assert pomio.getVoidIdx() not in np.unique(superPixelLabels) , "The set of predicted labels for the image contains the void label.  It shouldn't."
    
    return labelImage


def plotSuperPixelImage(sourceImage, labelledImage, orientation):
    print "\n*Now plotting source & labelled image for visual comparison."
    
    plt.interactive(1)
    plt.figure()
    
    pomio.showClassColours()
    plt.figure()
    
    print "*Unique labels from superpixel classification = ", np.unique(labelledImage)
    plt.subplot(1,2,1)
    plt.imshow(sourceImage, origin=orientation)
    
    plt.subplot(1,2,2)
    #pomio.showLabels(labelledImage)
    plt.imshow(labelledImage, origin=orientation)


def assignClassLabelToSuperPixel(superPixelValueMask, imagePixelLabels):
    """This function provides basic logic for setting the overall class label for a superpixel"""
    voidIdx = pomio.getVoidIdx()
    
    # just adopt the most frequently occurring class label as the superpixel label
    superPixelConstituentLabels = imagePixelLabels[superPixelValueMask]
    
    labelCount = stats.itemfreq(superPixelConstituentLabels)
    
    maxLabelFreq = np.max(labelCount[:, 1])
    maxLabelIdx = (labelCount[:,1] == maxLabelFreq)
    
    maxLabel = labelCount[maxLabelIdx]
    
    numSuperPixels = np.sum(labelCount[:,1])
    
    percentOfPixelsRequired = 0.5
    requiredPixelThreshold = np.round(0.5*numSuperPixels , 0).astype(int)
    
    maxLabelValue = voidIdx
    
    # what if the max count gives more than 1 match?  Naughty superpixel.
    # It would be nice to select the class that is least represented in the dataset so far...
    if np.shape(maxLabel)[0] > 1:
    
        for idx in range(0, np.shape(maxLabel)[0]):
            print "\t\tINFO:: max label:" , maxLabel[idx,:]
            
            if maxLabel[idx,0] != voidIdx and maxLabel[idx,1] >= requiredPixelThreshold:
                maxLabelValue = maxLabel[idx,0]
                break
            # If no match, maxLabelValues is equal to void, and will be skipped in training
                
    else:
        maxLabelValue = maxLabel[0,0]
    
    return maxLabelValue


def testClassifier(classifierFilename, case):
    assert case=="lena" or case=="car" , "case parameter should be lena or car"
    
    superPixelClassifier = pomio.unpickleObject(classifierFilename)
    print "\n*Loaded classifier [ " , type(superPixelClassifier) , "] from:" , classifierFilename
    
    image = None
    orientation = None
    
    if case == "car":
        print "*Loading MSRC car image::"
        image = pomio.msrc_loadImages("/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2", ['Images/7_3_s.bmp'] )[0].m_img
        orientation = "lower"

    elif case == "lena":
        print "*Loading Lena.jpg"
        image = skimage.data.lena()
        orientation = "upper"
    
    print "*Predicting superpixel labels in image::"
    numberSuperPixels = 400
    superPixelCompactness = 10
    [superPixelLabels, superPixelsMask] = predictSuperPixelLabels(superPixelClassifier, image,numberSuperPixels, superPixelCompactness)
    
    carSuperPixelLabels = getSuperPixelLabelledImage(image, superPixelsMask, superPixelLabels)
    
    plotSuperPixelImage(image, carSuperPixelLabels, orientation)

