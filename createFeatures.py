#!/usr/bin/env python
import argparse

"""
Command-line tool for creating pixel and super-pixel feature sets from images
"""

# Overhead example:
#
#  ./createFeatures.py ~/data/sceneLabelling/overhead/MSRC-like overhead --type pkl --scaleFrac 1.0 --splitRatio 1 0 0
#

parser = argparse.ArgumentParser(description='Create pixel or super-pixel features for MRF project.')

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
parser.add_argument('--ftype', type=str, default='classic', choices=['classic'],\
                      help = 'Feature type.' )
parser.add_argument('--aggtype', type=str, default='classic', \
                      choices=['classic'],\
                      help = 'Super-pixel feature aggregation type.' )
parser.add_argument('--nbCores', type=int, default=1, \
                        help='Number of cores to use in processing')
parser.add_argument('--v', action   = 'store_true')

# arguments
parser.add_argument('MSRCPath', type=str, action='store', \
                        help='file path of MSRC data (should have Images and GroundTruth below this dir)')
parser.add_argument('outfileBase', type=str, action='store', \
                        help='filename base for output files.  An output file will be created for each of training and test features and labels.')

args = parser.parse_args()

numberSuperPixels = args.nbSuperPixels
superPixelCompactness = args.superPixelCompactness

print 'Using %d cores' % args.nbCores

import pomio
import numpy as np
import superPixels
import features
import classification

# Function to take msrc data, create features and labels for superpixels and then save to disk
def createAndSaveFeatureLabelData(
  msrcData,
  outfileBase,
  dataType,
  outfileType,
  nbSuperPixels,
  superPixelCompactness,
  ftype, aggtype, verbose
  ):

    if verbose:
      print 'Creating feature set ', outfileBase

    if dataType == None or dataType == "":
        outfileFtrs = '%s_ftrs.%s' % ( outfileBase, outfileType )
        outfileLabs = '%s_labs.%s' % ( outfileBase, outfileType )
    else:
        outfileFtrs = '%s_%s_ftrs.%s' % ( outfileBase, dataType, outfileType )
        outfileLabs = '%s_%s_labs.%s' % ( outfileBase, dataType, outfileType )

    # Check can write these files.
    f=open(outfileFtrs,'w')
    f.close()
    f=open(outfileLabs,'w')
    f.close()

    if verbose:
      print '  - computing superpixels'
    allSuperPixels = superPixels.computeSuperPixelGraphMulti( \
      [ z.m_img for z in msrcData ],
      'slic',
      [nbSuperPixels, superPixelCompactness], nbCores=args.nbCores )
    if verbose:
      print '  - computing features'
    superPixelFeatures = features.computeSuperPixelFeaturesMulti(
      [z.m_img for z in msrcData], allSuperPixels, ftype, aggtype, asMatrix=True, nbCores=args.nbCores
      )
    if verbose:
      print '  - extracting labels'
    superPixelLabels = classification.computeSuperPixelLabelsMulti( \
      [z.m_gt for z in msrcData], allSuperPixels )

    assert np.all( np.isfinite( superPixelFeatures ) )

    # Don't save features with void labels.
    good = ( superPixelLabels != pomio.getVoidIdx() )
    if not np.all(good):
      if verbose:
        print '   - discarding %d superpixels with void labels' % np.count_nonzero( np.logical_not( good ) )
      superPixelLabels = superPixelLabels[good]
      superPixelFeatures = superPixelFeatures[good,:]

    if verbose:
      print '  - writing %d feature vectors of dimension %d to output files' % \
          (superPixelFeatures.shape[0], superPixelFeatures.shape[1])

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


# This has already been checked.  To be sure, to be sure
assert (trainSplit + cvSplit + testSplit) == 1.0 , "Data split values do not add up to 1 : " + str(trainSplit + cvSplit + testSplit) + ".  Check input data: " + str(splitRatio)
assert outfileType in ['csv', 'pkl'], 'Unknown outfile type ' + outfileType
assert 0 <= scaleFrac and scaleFrac <= 1

# Only write the type of data (i.e. train, validation, test) if a non-default split is provided as input
writeDataType = False
if (trainSplit == 1.0) and (cvSplit == 0.0 and testSplit == 0.0):
    writeDataType = False
else:
    # we get here if we have non-default splits
    writeDataType = True

# Do the expected thing if we have a non-default split
if args.v:
  print 'Loading data'

if writeDataType == True:
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
                                   superPixelCompactness,
                                   args.ftype, args.aggtype, args.v )

    if testData != None and len(testData)>0:
        print "Create & save test feature and label superpixel data"
        createAndSaveFeatureLabelData( testData,
                                       outfileBase,
                                       "test",
                                       outfileType,
                                       numberSuperPixels,
                                       superPixelCompactness,
                                       args.ftype, args.aggtype, args.v )
    if cvData != None and len(cvData) > 0:
        print "Create & save validation feature and label superpixel data"
        createAndSaveFeatureLabelData( cvData,
                                       outfileBase,
                                       "crossValid",
                                       outfileType,
                                       numberSuperPixels,
                                       superPixelCompactness,
                                       args.ftype, args.aggtype, args.v )

# Otherwise, just process all the data as a single data set, and write data to file without type in the filename
else:
    keepClassDistForTraining= False #!!
    [actualData, cvData, testData] = pomio.splitInputDataset_msrcData(\
        pomio.msrc_loadImages(msrcDataDirectory, subset=None),
        scaleFrac, keepClassDistForTraining, trainSplit , cvSplit , testSplit)

    assert actualData != None , "Training data object is null"
    createAndSaveFeatureLabelData( actualData,
                                   outfileBase,
                                   "",
                                   outfileType,
                                   numberSuperPixels,
                                   superPixelCompactness,
                                   args.ftype, args.aggtype, args.v )



