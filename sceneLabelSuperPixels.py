#!/usr/bin/env python

# Example:
#
#     ./sceneLabelSuperPixels.py randForestClassifierSP.pkl /home/jamie/data/MSRC_ObjCategImageDatabase_v2/Images/7_3_s.bmp

import pickle as pkl
import sys
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.ndimage.filters
import cython_uflow as uflow
import sklearn
import sklearn.ensemble
import bonzaClass
import amntools
import pomio
import FeatureGenerator
import slic
import SuperPixels
import argparse
import skimage

parser = argparse.ArgumentParser(description='Classify image and then apply MRF at the superPixel level.')
parser.add_argument('clfrFn', type=str, action='store', \
                        help='filename of pkl or csv superPixel classifier file')
parser.add_argument('--adjFn', type=str, action='store', \
                        help='filename of pkl or csv superPixel class adjacency probability matrix file')
parser.add_argument('infile', type=str, action='store', \
                        help='filename of input image to be classified')
parser.add_argument('--outfile', type=str, action='store', \
                        help='filename of output image with MRF inferred labels')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--K', type=float, action='store', default=0.1, \
                        help='Weighting for pairwise potential term in MRF.')
parser.add_argument('--nbrPotentialMethod', type=str, action='store', \
                        choices=['degreeSensitive', 'adjacencyAndDegreeSensitive'], default='degreeSensitive',\
                        help='Neighbour potential method.  If adjacency is used, then --adjFn must be specified.')

args = parser.parse_args()

K = args.K
dointeract = 1
dbgMode = 0


#
# MAIN
#
imgRGB = imread( args.infile )

# Turn image into superpixels.
spix = SuperPixels.computeSuperPixelGraph( imgRGB, 'slic', [400,10] )

print 'Loading classifier...'
clfr = bonzaClass.loadObject(args.clfrFn)
if args.adjFn != None and len(args.adjFn)>0:
    print 'Loading adjacency probs...'
    adjProbs = bonzaClass.loadObject(args.adjFn)
    # This is actually a bunch of counts.  Some will be zero, which is probably
    # a sampling error, so let's offset with some default number of counts.
    adjProbs += 10.0
    # Now turn it into normalised probabilities.
    # todo: hey but this is not normalised for default class probability!
    adjProbs /= adjProbs.sum()
    # transform
    adjProbs = -np.log( adjProbs )
else:
    adjProbs = None

# prefer to merge regions with high degree
if args.nbrPotentialMethod == 'adjacencyAndDegreeSensitive':
    assert adjProbs != None, 'You asked for neighbour potential method "%s", but no adjacency probs specified'\
        % args.nbrPotentialMethod

print 'Computing superpixel features...'
ftrs = FeatureGenerator.generateSuperPixelFeatures( imgRGB, spix.m_labels, [] )

print 'Computing class probabilities...'
classProbs = bonzaClass.classProbsOfFeatures(ftrs,clfr,\
                                                 requireAllClasses=False)

if args.verbose:
    plt.interactive(1)

    if adjProbs != None:
        plt.figure()
        plt.imshow(np.log(1+adjProbs), cmap=cm.get_cmap('gray'), interpolation='none')
        plt.title('Adjacency probabilities')
        plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(imgRGB)
    plt.title('original image')

    # show superpixels
    plt.figure()
    spix.draw()
    plt.title('Super Pixels')
    
    # show class labels per pixel
    classLabs = np.argmax( classProbs, 1 )
    # these are labs per region. turn into an image of labels.
    plt.figure()
    # +1 adds void class
    # todo: tidy this shemozzle up!
    pomio.showLabels( spix.imageFromSuperPixelData( classLabs + 1 ) )
    plt.title('Raw Classifier Labelling')
    plt.figure()
    pomio.showClassColours()
    
    plt.draw()
    if dointeract:
        print 'Click plot to continue...'
        plt.waitforbuttonpress()

#
# Do the inference
#


print 'Performing CRF inference...'
if args.verbose:
    plt.figure()

segResult = uflow.inferenceSuperPixel( \
    spix,\
    -np.log( np.maximum(1E-10, np.ascontiguousarray(classProbs) ) ), \
    adjProbs, \
    'abswap',\
    args.nbrPotentialMethod,\
    K )#, np.ascontiguousarray(nbrPotentialParams) )

# turn from label classes to msrc labels
segResult += 1
print '   done.'

if args.outfile and len(args.outfile)>0:
    print 'Writing output label file %s' % args.outfile
    outimg = pomio.msrc_convertLabelsToRGB( segResult )
    skimage.io.imsave(args.outfile, outimg)
    print '   done.'

# Show the result.
if args.verbose:
    pomio.showLabels(segResult)
    plt.title( 'Segmentation CRF result with K=%f' % K )
    plt.draw()
    print "labelling result, K = ", K 
    if dointeract:
        plt.waitforbuttonpress()

