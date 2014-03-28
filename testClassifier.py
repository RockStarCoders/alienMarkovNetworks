#!/usr/bin/env python
import argparse

# Usage:
#
#   testClassifier.py <clfr.pkl> <infile>
#
#
# Good test image:
#
#   cp ~/data/sceneLabelling/MSRC_ObjCategImageDatabase_v2/Images/3_7_s.bmp
#

parser = argparse.ArgumentParser(description='Apply superPixel classifier to an image.')
parser.add_argument('clfrFn', type=str, action='store', \
                        help='filename of pkl or csv superPixel classifier file')
parser.add_argument('infile', type=str, action='store', \
                        help='filename of input image to be classified')
parser.add_argument('--outfile', type=str, action='store', \
                        help='filename of output image.  This is an RGB image with the most-likely labelling at each pixel.')
parser.add_argument('--outprobsfile', type=str, \
                        help='filename of output probabilities image.  If specified, the 3D matrix of probabilities will be output as a pickle (.pkl) file')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nbSuperPixels', type=int, default=400, \
                        help='Desired number of super pixels in SLIC over-segmentation')
parser.add_argument('--superPixelCompactness', type=float, default=10.0, \
                        help='Super pixel compactness parameter for SLIC')
args = parser.parse_args()

import sys
import pomio
import sklearn.ensemble
import numpy as np
import matplotlib.pyplot as plt
import SuperPixelClassifier
import skimage

clfrFn = args.clfrFn
clfr = pomio.unpickleObject( clfrFn )

makeProbs = ( args.outprobsfile and len(args.outprobsfile)>0 )

#infile = args.infile
#outfile = args.outfile
#outprobsfile = args.outprobsfile
numberSuperPixels = args.nbSuperPixels
superPixelCompactness = args.superPixelCompactness

if args.verbose:
    plt.interactive(1)
    plt.figure()
    pomio.showClassColours()
    plt.figure()

print 'Classifying file ', args.infile
image = skimage.io.imread(args.infile)
[spClassPreds, spGraph, spClassProbs] = SuperPixelClassifier.predictSuperPixelLabels(clfr, image,numberSuperPixels, superPixelCompactness, makeProbs)
spClassPredsImage = spGraph.imageFromSuperPixelData( spClassPreds.reshape( (len(spClassPreds),1) ) )

if args.verbose:
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(args.infile)
    plt.subplot(1,2,2)
    pomio.showLabels(spClassPredsImage)
    plt.waitforbuttonpress()

if args.outfile and len(args.outfile)>0:
    print 'Writing output label file %s' % args.outfile
    outimg = pomio.msrc_convertLabelsToRGB( spClassPredsImage )
    skimage.io.imsave(args.outfile, outimg)
    print '   done.'

if makeProbs:
    print 'Writing output (superpixels, class probabilities) to pickle file %s' % \
        args.outprobsfile
    assert spClassProbs != None
    pomio.pickleObject( (spGraph,spClassProbs), args.outprobsfile )

if args.verbose:
    plt.interactive(0)
    plt.show()
