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

msrc_classToRGB = [\
('void'     , (0     ,0       ,0     )), \
('building' , (128   ,0       ,0     )), \
('grass'    , (0     ,128     ,0     )), \
('tree'     , (128   ,128     ,0     )), \
('cow'      , (0     ,0       ,128   )), \
('horse'    , (128   ,0       ,128   )), \
('sheep'    , (0     ,128     ,128   )), \
('sky'      , (128   ,128     ,128   )), \
('mountain' , (64    ,0       ,0     )), \
('aeroplane', (192   ,0       ,0     )), \
('water'    , (64    ,128     ,0     )), \
('face'     , (192   ,128     ,0     )), \
('car'      , (64    ,0       ,128   )), \
('bicycle'  , (192   ,0       ,128   )), \
('flower'   , (64    ,128     ,128   )), \
('sign'     , (192   ,128     ,128   )), \
('bird'     , (0     ,64      ,0     )), \
('book'     , (128   ,64      ,0     )), \
('chair'    , (0     ,192     ,0     )), \
('road'     , (128   ,64      ,128   )), \
('cat'      , (0     ,192     ,128   )), \
('dog'      , (128   ,192     ,128   )), \
('body'     , (64    ,64      ,0     )), \
('boat'     , (192   ,64      ,0     ))  \
]

msrc_classLabels = [z[0] for z in msrc_classToRGB]

def getNumClasses():
    return 24

def msrc_convertRGBToLabels( imgRGB ):
    imgL = 255 * np.ones( imgRGB.shape[0:2], dtype='uint8' )
    # For each label, find matching RGB and set that value
    for l,ctuple in enumerate(msrc_classToRGB):
        # Get a mask of matching pixels
        clr = ctuple[1]
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
    assert len(res) > 0
    return res




#
# Data preparation utils
#

def splitInputDataset_msrcData(msrcDataLocation, datasetScale=1.0 , keepClassDistForTraining=True, trainSplit=0.6, validationSplit=0.2, testSplit=0.2, ):
    assert (trainSplit + validationSplit + testSplit) == 1, "values for train, validation and test must sum to 1"
    # MUST ENSURE that the returned sample contains at least 1 example of each class label, or else the classifier doesn't consider all classes!
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to evaluate good regularisation parameter and test to evalaute generalised performance
    msrcImages = msrc_loadImages(msrcDataLocation)
    totalImages = np.shape(msrcImages)[0]
    totalSampledImages = np.round(totalImages * datasetScale , 0).astype('int')
    
    trainDataSize = np.round(totalSampledImages * trainSplit , 0)
    testDataSize = np.round(totalSampledImages * testSplit , 0)
    validDataSize = np.round(totalSampledImages * validationSplit , 0)
    
    # Get random samples from list
    if keepClassDistForTraining == False:
        
        trainData, msrcImages = selectRandomSetFromList(msrcImages, trainDataSize, True)
        testData, msrcImages = selectRandomSetFromList(msrcImages, testDataSize, False)
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = selectRandomSetFromList(msrcImages, validDataSize, False)
            
        # make sure all classes are included in training set (note void is processed out later)
        trainClasses = None
    
        for idx in range(0, len(trainData)):
            if trainClasses == None:
                trainClasses = np.unique(trainData[idx].m_gt)
            else:
                trainClasses = np.unique(np.append( trainClasses , np.unique(trainData[idx].m_gt) ) )
        
        numClasses = getNumClasses()
        # Note here we don't filter out void - do that at the pixel level when generating features
        assert np.size(trainClasses) == numClasses , "Training failed to include each of the 24 classes:: trainClasses = " + str(trainClasses)
        
        print "\nAssigned " + str(np.shape(trainData)) + " images to TRAIN set, " + str(np.shape(testData)) + " samples to TEST set and "+ str(np.shape(validationData)) + " samples to VALIDATION set"
        
        return [trainData, validationData, testData]
        
    elif keepClassDistForTraining == True:
        # Get class frequency count from image set
        classPixelCounts = PossumStats.totalPixelCountPerClass(msrcDataLocation, printTotals=False)[1]
        
        # normalise the frequency values to give ratios for each class
        sumClassCount = np.sum(classPixelCounts)
        classDist = classPixelCounts / float(sumClassCount)
        
        # use class sample function to create sample sets with same class ratios
        trainData, msrcImages = classSampleFromList(msrcImages, trainDataSize, classDist, True)
        testData, msrcImages = selectRandomSetFromList(msrcImages, testDataSize, True)
        
        if datasetScale == 1.0:
            validationData = msrcImages
        else:
            validationData, msrcImages = selectRandomSetFromList(msrcImages, validDataSize, True)
        
        
        # make sure all classes are included in training set (note void is processed out later)
        trainClasses = None
    
        for idx in range(0, len(trainData)):
            if trainClasses == None:
                trainClasses = np.unique(trainData[idx].m_gt)
            else:
                trainClasses = np.unique(np.append( trainClasses , np.unique(trainData[idx].m_gt) ) )
        
        numClasses = getNumClasses()
        assert np.size(trainClasses) == numClasses , "Training data does not include each of the 24 classes:: trainClasses = " + str(trainClasses)
        
        print "\nAssigned " + str(np.shape(trainData)) + " images to TRAIN set, " + str(np.shape(testData)) + " samples to TEST set and "+ str(np.shape(validationData)) + " samples to VALIDATION set"
        
        return [trainData, validationData, testData]


def classSampleFromList(msrcData , numberSamples, classDist, includeAllClassLabels=True):
    """This function takes random samples from input dataset (wihout replacement) maintaining a preset class label ratio.
    The indices of the are assumed to align with the indicies of the class labels.
    Returns a list of data samples and the reduced input dataset."""
    
    assert( np.size(classDist) == getNumClasses()) , "\n\tWARN:: For some reason the class distribution array doesnt have 24 elements - " + str(np.size(classDist))
    
    classSampleSizes = np.round((numberSamples * classDist) , 0).astype('int')
    
    result = []
    
    addedCount = 0
    if includeAllClassLabels == True:
        for idx in range(0, np.size(classSampleSizes)):
            if classSampleSizes[idx] == 0.0:
                classSampleSizes[idx] = 1
                addedCount = addedCount + 1
        numClassSamples = int(np.sum(classSampleSizes) )
        print "\tWARN: There were" , addedCount , "0 classes - each has been replaced by 1.  Returned sample size =" , numClassSamples
    
        # if number of required samples is less than number of class, over-rule
        if numberSamples < getNumClasses():
            print "\tWARN: You wanted all classes present, but requested #samples less than #classes.  Will return data including all classes."
            return selectRandomSetFromList(msrcData, getNumClasses(), includeAllClassLabels=True)
        else:
            print "\tINFO: Seek to maintain class dist AND include all classes"
            includedClasses = None
            for labelIdx in range(0 , np.size(classDist)):
                
                classSampleSize = classSampleSizes[labelIdx]
                classSampleCount = 0
                
                # Get samples and add to list while less than desired sample size for the class
                while classSampleCount < classSampleSize:
                    
                    sampleIdx = np.random.randint(0, np.size(msrcData))
                    sample = msrcData[sampleIdx]
                    
                    if labelIdx in sample.m_gt:
                        if includedClasses == None:
                            includedClasses = np.unique(sample.m_gt)
                            result.insert(np.size(result) , sample)
                            msrcData.pop(sampleIdx)
                            classSampleCount = classSampleCount+1
                            
                        else:
                            # add to included classes
                            imgClasses = np.unique(sample.m_gt)
                            includedClasses = np.unique(sample.m_gt)
                            result.insert(np.size(result) , sample)
                            msrcData.pop(sampleIdx)
                            classSampleCount = classSampleCount+1
                            # update included classes list
                            includedClasses = np.unique( np.append( includedClasses, imgClasses ) )
                    
            else:
                # Simply run through class samples list
                while classSampleCount < classSampleSize:
                    
                    sampleIdx = np.random.randint(0, np.size(msrcData))
                    sample = msrcData[sampleIdx]
                    
                    if labelIdx in sample.m_gt:
                        includedClasses = np.unique(sample.m_gt)
                        result.insert(np.size(result) , sample)
                        msrcData.pop(sampleIdx)
                        classSampleCount = classSampleCount+1
                        
    return result, msrcData
        


def selectRandomSetFromList(data, numberSamples, includeAllClassLabels):
    """This function randomly selects a number of samples from an array without replacement.
    Returns an array of the samples, and the resultant reduced data array."""
    idx = 0
    result = []
    if includeAllClassLabels == True:
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

