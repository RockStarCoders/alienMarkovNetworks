# Module for image and data i/o
import glob
import pylab
import numpy as np

import skimage.io

import PossumStats


# MSRC Image Segmentation database V2:
#
#   http://research.microsoft.com/en-us/projects/ObjectClassRecognition/
#
# It comes with a html doc file describing the labels rgb signatures.

msrc_classToRGB = {\
'void'     : (0     ,0       ,0     ), \
'building' : (128   ,0       ,0     ), \
'grass'    : (0     ,128     ,0     ), \
'tree'     : (128   ,128     ,0     ), \
'cow'      : (0     ,0       ,128   ), \
'horse'    : (128   ,0       ,128   ), \
'sheep'    : (0     ,128     ,128   ), \
'sky'      : (128   ,128     ,128   ), \
'mountain' : (64    ,0       ,0     ), \
'aeroplane': (192   ,0       ,0     ), \
'water'    : (64    ,128     ,0     ), \
'face'     : (192   ,128     ,0     ), \
'car'      : (64    ,0       ,128   ), \
'bicycle'  : (192   ,0       ,128   ), \
'flower'   : (64    ,128     ,128   ), \
'sign'     : (192   ,128     ,128   ), \
'bird'     : (0     ,64      ,0     ), \
'book'     : (128   ,64      ,0     ), \
'chair'    : (0     ,192     ,0     ), \
'road'     : (128   ,64      ,128   ), \
'cat'      : (0     ,192     ,128   ), \
'dog'      : (128   ,192     ,128   ), \
'body'     : (64    ,64      ,0     ), \
'boat'     : (192   ,64      ,0     )  \
}

msrc_classLabels = msrc_classToRGB.keys()

def getNumClasses():
    return 24

def msrc_convertRGBToLabels( imgRGB ):
    imgL = 255 * np.ones( imgRGB.shape[0:2], dtype='uint8' )
    # For each label, find matching RGB and set that value
    l = 0
    for lname,clr in msrc_classToRGB.items():
        # Get a mask of matching pixels
        msk = np.logical_and( imgRGB[:,:,0]==clr[0], \
                                  np.logical_and( imgRGB[:,:,1]==clr[1], \
                                                      imgRGB[:,:,2]==clr[2] ) )
        # Set these in the output image
        imgL[msk] = l
        l += 1
    # Check we got every pixel
    assert( not np.any( imgL == 255 ) )
    return imgL

class msrc_Image:
    'Structure containing image and ground truth from MSRC v2 data set'
    m_img = None
    m_gt  = None
    m_hq  = None
    m_imgFn = None
    m_gtFn  = None
    m_hqFn  = None

    def __init__( self,  fn, gtfn, hqfn ):
        # load the image (as numpy nd array, 8bit)
        self.m_img = pylab.imread( fn )
#         self.m_gt  = msrc_convertRGBToLabels( pylab.imread( gtfn ) )
        self.m_gt  = msrc_convertRGBToLabels( skimage.io.imread( gtfn ) )
        # not necessarily hq
        try:
            #self.m_hq  = msrc_convertRGBToLabels( pylab.imread( hqfn ) )
            self.m_hq  = msrc_convertRGBToLabels( skimage.io.imread( hqfn ) )
        except IOError:
            self.m_hq = None
        self.m_imgFn = fn
        self.m_gtFn  = gtfn
        self.m_hqFn  = hqfn
   
# dataSetPath is the base directory for the data set (subdirs are under this)
# Returns a list of msrc_image objects.
def msrc_loadImages( dataSetPath ):
    res = []
    # For each image file:
    for fn in glob.glob( dataSetPath + '/Images/*.bmp' ):
        # load the ground truth, convert to discrete label
        gtfn = fn.replace('Images/', 'GroundTruth/').replace('.bmp','_GT.bmp')
        hqfn = fn.replace('Images/', 'SegmentationsGTHighQuality/').replace('.bmp','_HQGT.bmp')
        # create an image object, stuff in list
        res.append( msrc_Image( fn, gtfn, hqfn ) )
        #break
    return res




#
# Data preparation utils
#

def splitInputDataset_msrcData(msrcDataLocation, datasetScale=1.0 , keepClassDist=True, trainSplit=0.6, validationSplit=0.2, testSplit=0.2, ):
    assert (trainSplit + validationSplit + testSplit) == 1, "values for train, validation and test must sum to 1"
    # MUST ENSURE that the returned sample contains at least 1 example of each class label, or else the classifier doesn't consider all classes!
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to find good regularisation parameters and test for performance
    print "Loading images from msrc dataset"
    msrcImages = msrc_loadImages(msrcDataLocation)
    print "Completed loading"
    
    totalImages = np.shape(msrcImages)[0]
    
    totalSampledImages = np.round(totalImages * datasetScale , 0).astype('int')
    
    trainDataSize = np.round(totalSampledImages * trainSplit , 0)
    testDataSize = np.round(totalSampledImages * testSplit , 0)
    validDataSize = np.round(totalSampledImages * validationSplit , 0)
    
    # Get random samples from list
    if keepClassDist == False:
        
        trainData, msrcImages = selectRandomSetFromList(msrcImages, trainDataSize, True)
        testData, msrcImages = selectRandomSetFromList(msrcImages, testDataSize, False)
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = selectRandomSetFromList(msrcImages, validDataSize, False)
            
        print "\nRandomly assigned " + str(np.shape(trainData)) + " subset of msrc data to TRAIN set"
        print "Randomly assigned " + str(np.shape(testData)) + " subset of msrc data to TEST set"
        print "Randomly assigned " + str(np.shape(validationData)) + " msrc data to VALIDATION set"
        
        # make sure all classes are included in training set (note void is processed out later)
        trainClasses = None
    
        for idx in range(0, len(trainData)):
            if trainClasses == None:
                trainClasses = np.unique(trainData[idx].m_gt)
            else:
                trainClasses = np.unique(np.append( trainClasses , np.unique(trainData[idx].m_gt) ) )
        
        numClasses = getNumClasses()
        assert np.size(trainClasses) == numClasses , "Training failed to include each of the 24 classes:: trainClasses = " + str(trainClasses)
        
        return [trainData, validationData, testData]
        
    elif keepClassDist == True:
        # Get class frequency count from image set
        classPixelCounts = PossumStats.totalPixelCountPerClass(msrcDataLocation, printTotals=False)[1]
        
        # normalise the frequency values to give ratios for each class
        maxClassCount = np.max(classPixelCounts)
        
        classDist = classPixelCounts / maxClassCount
        
        # use class sample function to create sample sets with same class ratios
        trainData, msrcImages = classSampleFromList(msrcImages, trainDataSize, classDist, True)
        testData, msrcImages = classSampleFromList(msrcImages, testDataSize, classDist, False)
        
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = classSampleFromList(msrcImages, validDataSize, classDist, False)
        
        
        # make sure all classes are included in training set (note void is processed out later)
        trainClasses = None
    
        for idx in range(0, len(trainData)):
            if trainClasses == None:
                trainClasses = np.unique(trainData[idx].m_gt)
            else:
                trainClasses = np.unique(np.append( trainClasses , np.unique(trainData[idx].m_gt) ) )
        
        numClasses = getNumClasses()
        assert np.size(trainClasses) == numClasses , "Training failed to include each of the 24 classes:: trainClasses = " + str(trainClasses)
        
        print "\nAssigned " + str(np.shape(trainData)) + " randomly selected, class ratio preserved samples subset of msrc data to TRAIN set"
        print "Assigned " + str(np.shape(testData)) + " randomly selected, class ratio preserved samples subset of msrc data to TEST set"
        print "Assigned " + str(np.shape(validationData)) + " randomly selected, class ratio preserved samples subset of msrc data to VALIDATION set"
        
        return [trainData, validationData, testData]


def classSampleFromList(msrcData , numberSamples, classDist, keepZeroClasses=True):
    """This function takes random samples from input dataset (wihout replacement) maintaining a preset class label ratio.
    The indices of the are assumed to align with the indicies of the class labels.
    Returns a list of data samples and the reduced input dataset."""
    
    classSampleSizes = np.round((numberSamples * classDist) , 0).astype('int')
    
    addedCount = 0
    
    if keepZeroClasses == True:
        print "\t Warn: Changing class sample 0 proportions to 1, before normalising"
        for idx in range(0, np.size(classSampleSizes)):
            if classSampleSizes[idx] == 0:
                classSampleSizes[idx] = 1
                addedCount = addedCount + 1
                
        # need to see how many zeros, and either edit the dist or throw an error
    
    classSum = int(np.sum(classSampleSizes) )
    assert (numberSamples == classSum-addedCount) , "Some rounding error on total samples versus samples by class" + str(classSum) + " , " + str(addedCount) + str(numberSamples)
    
    sampleResult = []
    
    
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
            sampleIdx = np.random.randint(0, np.size(msrcData))
            sample = msrcData[sampleIdx]
            
            # check to see if image contains the required index
            if labelIdx in sample.m_gt:
                # if so, add to the sample list and increment counters
                sampleResult.insert( np.size(sampleResult) , sample  )
                
                msrcData.pop(sampleIdx)
                classSampleCount = classSampleCount + 1
                sampleCount = sampleCount + 1
    
    return sampleResult, msrcData


def selectRandomSetFromList(data, numberSamples, includeAllClassLabels):
    """This function randomly selects a number of samples from an array without replacement.
    Returns an array of the samples, and the resultant reduced data array."""
    idx = 0
    result = []
    if includeAllClassLabels == True:
        print "Will attempt to find samples that include at least one example of all 24 classes"
        includedClasses = None
        
        while idx < numberSamples:
            # randomly sample from imageset, and assign to sample array
            # randIdx = np.round(random.randrange(0,numImages), 0).astype(int)
            randIdx = np.random.randint(0, np.size(data))
            sample = data[randIdx]
            
            # make assumption we have labeled MSRC data :)
            if includedClasses == None:
                # We need to start somewhere
                includedClasses = np.unique(sample.m_gt)
                result.insert(idx, sample)
                # now remove the image from the dataset to avoid duplication
                data.pop(randIdx)
                idx = idx+1
                
            else:
                # check if the sample offers new classes, if not break and pick a new sample
                imgClasses = np.unique(sample.m_gt)
                
                if (includedClasses != None) and (np.size(includedClasses) < getNumClasses()):
                    #print "Comparing image classes::\n\t" , imgClasses, "\nwith classes in the data sample so far::\n\t" , includedClasses
                    
                    newElements = np.size(np.setdiff1d(imgClasses , includedClasses) )
    
                    if (newElements > 0):
                        
                        # we have new class labels so add to data
                        result.insert(idx, sample)
                        # now remove the image from the dataset to avoid duplication
                        data.pop(randIdx)
                        idx = idx+1
                        # update included classes list
                        includedClasses = np.unique( np.append( includedClasses, np.unique(sample.m_gt) ) )
                    
                else:
                    # we've got all the classes, just grab random samples if we still need more
                    result.insert(idx, sample)
                    # now remove the image from the dataset to avoid duplication
                    data.pop(randIdx)
                    idx = idx+1
    
    elif includeAllClassLabels == False:
        while idx < numberSamples:
            # randomly sample from imageset, and assign to sample array
            # randIdx = np.round(random.randrange(0,numImages), 0).astype(int)
            randIdx = np.random.randint(0, np.size(data))
            result.insert(idx, data[randIdx])
            # now remove the image from the dataset to avoid duplication
            data.pop(randIdx)
            idx = idx+1
        
    return result, data

