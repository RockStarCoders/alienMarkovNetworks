#!/usr/bin/env python
import sys
import pomio
import sklearn.ensemble
import numpy as np
import matplotlib.pyplot as plt
import SuperPixelClassifier
import skimage
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
                        help='filename of output image to be classified')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

clfrFn = args.clfrFn
clfr = pomio.unpickleObject( clfrFn )

infile = args.infile
outfile = args.outfile

if args.verbose:
    plt.interactive(1)
    plt.figure()
    pomio.showClassColours()
    plt.figure()

print 'Classifying file ', args.infile
image = skimage.io.imread(args.infile)
[spClassPreds, spGraph] = SuperPixelClassifier.predictSuperPixelLabels(clfr, image)
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

if args.verbose:
    plt.interactive(0)
    plt.show()
