#!/usr/bin/env python
import sys
import SuperPixelClassifier
import pomio
import numpy as np

# Usage:
#
#   createFeatures.py <MSRC data path> <scaleFrac> <splitRatio> <outfileBase> <type=csv|pkl>
#


# Function to take msrc data, create features and labels for superpixels and then save to disk
def createAndSaveFeatureLabelData(msrcData, outfileBase, dataType, outfileType):
    outfileFtrs = '%s_%s_ftrs.%s' % ( outfileBase, dataType, outfileType )
    outfileLabs = '%s_%s_labs.%s' % ( outfileBase, dataType, outfileType )
    # Check can write these files.
    f=open(outfileFtrs,'w')
    f.close()
    f=open(outfileLabs,'w')
    f.close()
    
    # Edited getSuperPixelTrainingData to allow user-defined data split ratios
    superPixelData     = SuperPixelClassifier.getSuperPixelData(trainData)
    
    superPixelFeatures = superPixelTrainData[0]
    superPixelLabels   = superPixelTrainData[1].astype(np.int32)
    
    assert np.all( np.isfinite( superPixelFeatures ) )
    
    # Output
    if outfileType == 'pkl':
        pomio.pickleObject( superPixelFeatures, outfileFtrs )
        pomio.pickleObject( superPixelLabels,   outfileLabs )
    elif outfileType == 'csv':
        pomio.writeMatToCSV( superPixelFeatures, outfileFtrs )
        pomio.writeMatToCSV( superPixelLabels,   outfileLabs )
    else:
        assert False, 'unknown output file format ' + outfileType

    print 'Output written to file ', outfileFtrs, ' and ', outfileLabs


msrcDataDirectory = sys.argv[1]
scaleFrac         = float(sys.argv[2]) # We aren't using this, assume we want all images
splitRatio        = sys.argv[3]
outfileBase       = sys.argv[4]
outfileType       = sys.argv[5]

trainSplit = 0.0
cvSplit = 0.0
testSplit = 0.0

ratios = splitRatio.split(",")

assert (len(ratios) == 2 or len(ratios) == 3) , "You did not specifiy 2 or three data split ratios.  Check input value: " + str(splitRatio)

# parse the ratio string, test it matches the number of splits
if(len(ratios) == 2):
    trainSplit = float(ratios[0])
    testSplit = float(ratios[1])
    cvSplit = 0.0
elif(len(ratios) == 3):
    trainSplit = float(ratios[0])
    cvSplit = float(ratios[1])
    testSplit = float(ratios[2])

assert (trainSplit + cvSplit + testSplit) == 1.0 , "Data split values do not add up to 1 : " + str(trainSplit + cvSplit + testSplit) + ".  Check input data: " + str(splitRatio) + " & " + str(ratios)
assert outfileType in ['csv', 'pkl'], 'Unknown outfile type ' + outfileType
assert 0 <= scaleFrac and scaleFrac <= 1


# Get train, test and cv datasets
print "\nSplitting data into sets: train =" , trainSplit , "test =" , testSplit , "cvSplit =" , cvSplit
[trainData, cvData, testData] = pomio.splitInputDataset_msrcData(pomio.msrc_loadImages(msrcDataDirectory, subset=None) , 1.0 , True, trainSplit , cvSplit , testSplit)

assert trainData != None , "Training data object is null"
assert len(trainData) > 0 , "Training data contains no data"

assert testData != None , "Testing data object is null"
assert len(testData) > 0 , "Testing data contains no data"


# Process and persist feature and labels for superpixels in image dataset
print "Create & save training feature and label superpixel data"
createAndSaveFeatureLabelData(trainData, 1.0, "train" , outfileType)

print "Create & save test feature and label superpixel data"
createAndSaveFeatureLabelData(testData, 1.0, "test" , outfileType)
if(cvData != None and len(cvData) > 0):
    print ""
    createAndSaveFeatureLabelData(cvData, 1.0, "crossValid" , outfileType)




