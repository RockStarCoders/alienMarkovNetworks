# Module for image and data i/o
import glob
import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

import skimage.io

import PossumStats

# MSRC Image Segmentation database V2:
#
#   http://research.microsoft.com/en-us/projects/ObjectClassRecognition/
#
# It comes with a html doc file describing the labels rgb signatures.

msrc_classToRGB = [\
('void'     , (0     ,0       ,0     )), # 0 \
('building' , (128   ,0       ,0     )), # 1 \
('grass'    , (0     ,128     ,0     )), # 2 \
('tree'     , (128   ,128     ,0     )), # 3 \
('cow'      , (0     ,0       ,128   )), # 4 \
('horse'    , (128   ,0       ,128   )), # 5 \
('sheep'    , (0     ,128     ,128   )), # 6 \
('sky'      , (128   ,128     ,128   )), # 7 \
('mountain' , (64    ,0       ,0     )), # 8 \
('aeroplane', (192   ,0       ,0     )), # 9 \
('water'    , (64    ,128     ,0     )), # 10 \
('face'     , (192   ,128     ,0     )), # 11 \
('car'      , (64    ,0       ,128   )), # 12 \
('bicycle'  , (192   ,0       ,128   )), # 13 \
('flower'   , (64    ,128     ,128   )), # 14 \
('sign'     , (192   ,128     ,128   )), # 15 \
('bird'     , (0     ,64      ,0     )), # 16 \
('book'     , (128   ,64      ,0     )), # 17 \
('chair'    , (0     ,192     ,0     )), # 18 \
('road'     , (128   ,64      ,128   )), # 19 \
('cat'      , (0     ,192     ,128   )), # 20 \
('dog'      , (128   ,192     ,128   )), # 21 \
('body'     , (64    ,64      ,0     )), # 22 \
('boat'     , (192   ,64      ,0     ))  # 23 \
]

msrc_classLabels = [z[0] for z in msrc_classToRGB]

overhead_classToRGB = [\
('void'     , (0     ,0       ,0     )), # 0 \
('building' , (255   ,0       ,0     )), # 1 \
('grass'    , (0     ,255     ,0     )), # 2 \
('tree'     , (200   ,100     ,20    )), # 3 \
('water'    , (0     ,0       ,255   )), # 4 \
('road'     , (100   ,100     ,100   )), # 5 \
]

overhead_classLabels = [z[0] for z in overhead_classToRGB]

def getVoidIdx():
    return 0

def getNumLabels():
    # includes void for display purposes
    return len(msrc_classLabels)

def getNumClasses():
    # doesn't include void
    return len(msrc_classLabels)-1

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
    #plt.imshow(imgL==255)
    #plt.show()
    dodgyMsk = (imgL==255)
    if np.any( dodgyMsk ):
        print '  WARNING: there are %d pixels with invalid colours.  Setting these to void.' % dodgyMsk.sum()
        imgL[ dodgyMsk ] = 0
    return imgL

class msrc_Image:
    'Structure containing image and ground truth from MSRC v2 data set'
    m_img = None
    m_gt  = None
    m_hq  = None
    m_imgFn = None
    m_gtFn  = None
    m_hqFn  = None

    def __init__( self,  fn, gtfn, hqfn, verbose=False ):
        # load the image (as numpy nd array, 8bit)
        if verbose:
            print 'Loading image ', fn
        self.m_img = skimage.io.imread( fn ) #pylab.imread( fn )
        if verbose:
            print 'Loading gt image ', gtfn
        self.m_gt  = msrc_convertRGBToLabels( skimage.io.imread( gtfn ) )
        # not necessarily hq
        try:
            #self.m_hq  = msrc_convertRGBToLabels( pylab.imread( hqfn ) )
            if verbose:
                print 'Looking for high quality gt image ', hqfn
            self.m_hq  = msrc_convertRGBToLabels( skimage.io.imread( hqfn ) )
            if verbose:
                print '   - found'
        except IOError:
            if verbose:
                print '   - not found'
            self.m_hq = None
        self.m_imgFn = fn
        self.m_gtFn  = gtfn
        self.m_hqFn  = hqfn
   
# dataSetPath is the base directory for the data set (subdirs are under this)
# Returns a list of msrc_image objects.  subset should be a list of 
# filenames of the original image, relative to dataSetPath.
def msrc_loadImages( dataSetPath, subset=None ):
    res = []
    if subset == None:
        subset = glob.glob( dataSetPath + '/Images/*.bmp' ) + glob.glob( dataSetPath + '/Images/*.png' )
    else:
        subset = [ dataSetPath + '/' + fn for fn in subset ]
    # For each image file:
    for fn in subset:
        # load the ground truth, convert to discrete label
        gtfn = fn.replace('Images/', 'GroundTruth/').replace('.','_GT.')
        hqfn = fn.replace('Images/', 'SegmentationsGTHighQuality/').replace('.','_HQGT.')
        # create an image object, stuff in list
        res.append( msrc_Image( fn, gtfn, hqfn, verbose=True ) )
        #break
    assert len(res) > 0, 'zarro images loaded.  subset = %s' % subset
    return res




#
# Data preparation utils
#

def splitInputDataset_msrcData( msrcImages,
                                datasetScale=1.0,
                                keepClassDistForTraining=True,
                                trainSplit=0.6,
                                validationSplit=0.2,
                                testSplit=0.2
                                ):
    assert (trainSplit + validationSplit + testSplit) == 1, "values for train, validation and test must sum to 1"
    # MUST ENSURE that the returned sample contains at least 1 example of each class label, or else the classifier doesn't consider all classes!
    # go for a random 60, 20, 20 split for train, validate and test
    # use train to build model, validate to evaluate good regularisation parameter and test to evalaute generalised performance
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
        trainClasses = set()
    
        for idx in range(0, len(trainData)):
            trainClasses.update( trainData[idx].m_gt.flatten() )
        # Take void out of the set
        trainClasses.remove(0)

        numClasses = getNumClasses()
        # Note here we don't filter out void - do that at the pixel level when generating features
        if not len(trainClasses) == numClasses:
            print "  WARNING: Training failed to include each of the 24 classes:: trainClasses = " + str(trainClasses) + '. proceeding anyway since keepClassDistForTraining was false.'
        
        print "\nAssigned " + str(np.shape(trainData)) + " images to TRAIN set, " + str(np.shape(testData)) + " samples to TEST set and "+ str(np.shape(validationData)) + " samples to VALIDATION set"
        
        return [trainData, validationData, testData]
        
    elif keepClassDistForTraining == True:
        # Get class frequency count from image set
        classPixelCounts = PossumStats.totalPixelCountPerClass(msrcImages, printTotals=False)[1]
        
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
        
        
        # make sure all classes are included in training set
        trainClasses = set()
    
        for idx in range(0, len(trainData)):
            trainClasses.update( trainData[idx].m_gt.flatten() )
        trainClasses.remove(0)

        numClasses = getNumClasses()
        #print "\tINFO: number of labels=" , numLabels, " , number labels in training set=" , np.shape(trainClasses)[0] 
        assert len(trainClasses) == numClasses , "Training data does not include each label:: trainClasses = " + str(len(trainClasses)) + " vs " + str(numClasses)
        
        print "\nAssigned " + str(np.shape(trainData)) + " images to TRAIN set, " + str(np.shape(testData)) + " samples to TEST set and "+ str(np.shape(validationData)) + " samples to VALIDATION set"
        
        return [trainData, validationData, testData]


def classSampleFromList(msrcData , numberSamples, classDist, includeAllClassLabels=True):
    """This function takes random samples from input dataset (wihout replacement).
    Seeks to maintain user-specified preset class distribution, and also include all labels.
    The indices of the are assumed to align with the indicies of the class labels.
    Returns a list of data samples and the reduced input dataset."""
    
    numLabels = getNumLabels()
    #print "\tINFO: number of labels=" , numLabels, " , number labels in training=" , np.shape(classDist)[0]
    assert( np.size(classDist) == numLabels) , "\n\tWARN:: For some reason the labels in distribution array doesnt match label set - " + str(np.size(classDist)) + " vs. " + str(numLabels)

    if numberSamples == 0:
        return msrcData
    elif numberSamples == 1:
        print "\tWARN: You only asked for 1 sample, so will just return random sample - regardless of label options"
        return selectRandomSetFromList(msrcData, numberSamples, includeAllClassLabels=False) 

    # Would be nice to have general function to calc the min number of samples required to get all classes...
    
    # TODO: there is a bug here.  If numberSamples is 1 for example, all these are 0.
    classSampleSizes = np.round((numberSamples * classDist) , 0).astype('int')
    addedCount = 0
    for idx in range(0, np.size(classSampleSizes)):
        if classSampleSizes[idx] == 0 or classSampleSizes[idx] == 0.0:
            classSampleSizes[idx] = 1
            addedCount = addedCount + 1
            numClassSamples = int(np.sum(classSampleSizes) )
    print "\tWARN: There were" , addedCount , "classes with 0 samples.  Each has been replaced by 1 sample requirement::" , numClassSamples

    
    if includeAllClassLabels == True:
        # if number of required samples is less than number of class, over-rule
        if numberSamples > 1 and numberSamples < getNumClasses():
            print "\tWARN: You wanted all classes present, but requested #samples less than #classes; generate sample list including all classes."
            return selectRandomSetFromList(msrcData, getNumClasses(), includeAllClassLabels=True)

        else:
            print "\tINFO: Seek to maintain class dist AND include all classes"
            result = []
            includedClasses = None

            for labelIdx in range(0 , np.size(classDist)):
                
                classSampleSize = classSampleSizes[labelIdx]
                classSampleCount = 0
                
                # Add sample to list while we need labels to preserve class dist
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
            # Now return result
            return result, msrcData
                    
    else:
        # Simply run grab random samples, regardless of classes in samples
        return selectRandomSetFromList(msrcData, numberSamples, includeAllClassLabels=False)
        


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
        idx = 0
        while idx < numberSamples:
            # randomly sample from imageset, and assign to sample array
            # randIdx = np.round(random.randrange(0,numImages), 0).astype(int)
            randIdx = np.random.randint(0, np.size(data))
            result.insert(idx, data[randIdx])
            # now remove the image from the dataset to avoid duplication
            data.pop(randIdx)
            idx = idx+1
        
    return result, data


def showLabels( labimg ):
    clrs = [[z/255.0 for z in c[1]] for c in msrc_classToRGB]
    plt.imshow( labimg,\
                cmap=matplotlib.colors.ListedColormap(clrs),\
                vmin=0,\
                vmax=len(clrs)-1,\
                interpolation = 'none' )

def showClassColours():
    plt.clf()
    showLabels( np.array( [ np.arange(len(msrc_classLabels)) ] ).transpose() )
    plt.yticks( np.arange( len(msrc_classLabels) ), msrc_classLabels )


def writeMatToCSV(obj, outfile):
    assert type(obj) == np.ndarray
    f = open(outfile, 'w')
    if obj.dtype == np.int32:
        np.savetxt(f, obj, fmt='%d', delimiter=',')
    else:
        np.savetxt(f, obj, fmt='%0.8f', delimiter=',')
    f.close()

def readMatFromCSV( infile ):
    f = open( infile, 'r' )
    res = np.loadtxt(infile, delimiter=',')
    f.close()
    return res

def pickleObject(obj, fullFilename):
    if fullFilename.endswith(".pkl"):
        f = open( fullFilename , "w")
        pickle.dump(obj, f , True)
        f.close()
    else:
        print "Input filename did not end in .pkl - adding .pkl to filename."
        fullFilename= str(fullFilename)+".pkl"
        f = open( fullFilename , "w")
        pickle.dump(obj, f , True)
        f.close()
        
def unpickleObject(fullFilename):
    f = open(fullFilename, "r")
    object = pickle.load(f)
    f.close()
    return object


def readEvaluationListFromCsv(evalListFile):
# lines = pomio.readEvaluationListFromCsv("/home/amb/dev/mrf/data/eval/evalList.csv")

    assert evalListFile.endswith(".csv") , "Please provide a CSV list of prediction and ground truth files"
    f = open(evalListFile, "r")
    lines = f.read().splitlines()
    
    data = []
    
    for idx in range(0, np.shape(lines)[0]):
        data.append(lines[idx].split(","))
    
    return data

