#!/usr/bin/env python
import SuperPixelClassifier
import pomio
import numpy as np
import argparse


# Overhead example:
#
#  ./createFeatures.py ~/data/sceneLabelling/overhead/MSRC-like overhead --type pkl --scaleFrac 1.0 --splitRatio 1 0 0
#

parser = argparse.ArgumentParser(description='Create super-pixel features for MRF project.')

# options
parser.add_argument('--type', type=str, action='store', default='pkl', \
                        choices = ['pkl', 'csv'], \
                        help='output file type')
parser.add_argument('--scaleFrac', type=float, action='store', default=1.0, \
                        help='Fraction of the available data to use.' )
parser.add_argument('--splitRatio', action='store', default=[1.0,0.0,0.0], \
                        nargs=3, metavar=('rTrain','rValidation', 'rTest'),\
                        help = 'Ratio of data to use for training, validation and test. They don''t have to sum to unity.' )
parser.add_argument('--nbSuperPixels', type=int, default=400, \
                        help='Desired number of super pixels in SLIC over-segmentation')
parser.add_argument('--superPixelCompactness', type=float, default=10.0, \
                        help='Super pixel compactness parameter for SLIC')

# arguments
parser.add_argument('MSRCPath', type=str, action='store', \
                        help='file path of MSRC data (should have Images and GroundTruth below this dir)')
parser.add_argument('outfileBase', type=str, action='store', \
                        help='filename base for output files.  An output file will be created for each of training and test features, labels and adjacency probabilities.')

args = parser.parse_args()

numberSuperPixels = args.nbSuperPixels
superPixelCompactness = args.superPixelCompactness

# Function to take msrc data, create features and labels for superpixels and then save to disk
def createAndSaveFeatureLabelData(
      msrcData, outfileBase, dataType, outfileType, nbSuperPixels, superPixelCompactness
    ):
    outfileFtrs = '%s_%s_ftrs.%s' % ( outfileBase, dataType, outfileType )
    outfileLabs = '%s_%s_labs.%s' % ( outfileBase, dataType, outfileType )
    outfileAdj  = '%s_%s_adj.%s'  % ( outfileBase, dataType, outfileType )
    # Check can write these files.
    f=open(outfileFtrs,'w')
    f.close()
    f=open(outfileLabs,'w')
    f.close()
    
    # Edited getSuperPixelTrainingData to allow user-defined data split ratios
    superPixelData     = SuperPixelClassifier.getSuperPixelData(msrcData,nbSuperPixels,superPixelCompactness)
    
    superPixelFeatures = superPixelData[0]
    superPixelLabels   = superPixelData[1].astype(np.int32)
    superPixelClassAdj = superPixelData[2]

    assert np.all( np.isfinite( superPixelFeatures ) )

    #print superPixelClassAdj

    # Output
    if outfileType == 'pkl':
        pomio.pickleObject( superPixelFeatures, outfileFtrs )
        pomio.pickleObject( superPixelLabels,   outfileLabs )
        pomio.pickleObject( superPixelClassAdj, outfileAdj )
    elif outfileType == 'csv':
        pomio.writeMatToCSV( superPixelFeatures, outfileFtrs )
        pomio.writeMatToCSV( superPixelLabels,   outfileLabs )
        pomio.writeMatToCSV( superPixelClassAdj, outfileAdj )
    else:
        assert False, 'unknown output file format ' + outfileType

    print 'Output written to file ', outfileFtrs, ' and ', outfileLabs, ' and ', outfileAdj


msrcDataDirectory = args.MSRCPath
scaleFrac         = args.scaleFrac
# comes in as a list of strings
splitRatio        = np.array([float(z) for z in args.splitRatio])
splitRatio /= splitRatio.sum()
outfileBase       = args.outfileBase
outfileType       = args.type

trainSplit = float(splitRatio[0])
cvSplit = float(splitRatio[1])
testSplit = float(splitRatio[2])

# This has already been checked.  to be sure, to be sure
assert (trainSplit + cvSplit + testSplit) == 1.0 , "Data split values do not add up to 1 : " + str(trainSplit + cvSplit + testSplit) + ".  Check input data: " + str(splitRatio)
assert outfileType in ['csv', 'pkl'], 'Unknown outfile type ' + outfileType
assert 0 <= scaleFrac and scaleFrac <= 1


# Get train, test and cv datasets
print "\nSplitting data into sets: train =" , trainSplit , "test =" , testSplit , "cvSplit =" , cvSplit
keepClassDistForTraining= False #!!
[trainData, cvData, testData] = pomio.splitInputDataset_msrcData(\
    pomio.msrc_loadImages(msrcDataDirectory, subset=None),
    scaleFrac, keepClassDistForTraining, trainSplit , cvSplit , testSplit)

assert trainData != None , "Training data object is null"
assert len(trainData) > 0 , "Training data contains no data"

if testData == None or  len(testData) == 0:
    print 'WARNING: Testing data contains no data'

print "Creating training set from %d images, and test set from %d images" % (len(trainData), len(testData))

# Process and persist feature and labels for superpixels in image dataset
print "Create & save training feature and label superpixel data"
createAndSaveFeatureLabelData( trainData,
                               outfileBase,
                               "train",
                               outfileType,
                               numberSuperPixels,
                               superPixelCompactness )

if testData != None and len(testData)>0:
    print "Create & save test feature and label superpixel data"
    createAndSaveFeatureLabelData( testData,
                                   outfileBase,
                                   "test",
                                   outfileType,
                                   numberSuperPixels,
                                   superPixelCompactness )
if(cvData != None and len(cvData) > 0):
    print "Create & save validation feature and label superpixel data"
    createAndSaveFeatureLabelData( cvData,
                                   outfileBase,
                                   "crossValid",
                                   outfileType,
                                   numberSuperPixels,
                                   superPixelCompactness )




