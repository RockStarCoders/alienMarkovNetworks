#!/usr/bin/env python

"""
Command-line utility to do N-class super-pixel MRF segmentation.
"""

# Example:
#
#     ./sceneLabelSuperPixels.py randForestClassifierSP.pkl /home/jamie/data/MSRC_ObjCategImageDatabase_v2/Images/7_3_s.bmp

import argparse

parser = argparse.ArgumentParser(description='Classify image and then apply MRF at the superPixel level.')
parser.add_argument('--clfrFn', type=str, action='store', \
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
parser.add_argument('--nbSuperPixels', type=int, default=400, \
                        help='Desired number of super pixels in SLIC over-segmentation')
parser.add_argument('--superPixelCompactness', type=float, default=10.0, \
                        help='Super pixel compactness parameter for SLIC')

args = parser.parse_args()



import pickle as pkl
import sys
import numpy as np
import scipy
import scipy.io
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.ndimage.filters
import cython_uflow as uflow
import sklearn
import sklearn.ensemble
import amntools
import pomio
import FeatureGenerator
import slic
import superPixels
import skimage




def getAdjProbs(name):
    if name != None and len(name)>0:
        print 'Loading adjacency probs...'
        adjProbs = pomio.unpickleObject(name)
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
    return adjProbs

# prefer to merge regions with high degree
if args.nbrPotentialMethod == 'adjacencyAndDegreeSensitive':
    assert args.adjFn != None, 'You asked for neighbour potential method "%s", but no adjacency probs specified'\
        % args.nbrPotentialMethod


#
# MAIN
#
dointeract = 1
dbgMode = 0

# Class vars
K = args.K
spix = None
classProbs = None
adjProbs = None

precomputedMode = args.infile.endswith('.pkl') or args.infile.endswith('.mat')

if precomputedMode == True:
    # If the input file is a pkl (not image) assume we've run necessary superpixel classifier, and input is class label probabilities
    # Input is assumed to be a tuple [superpixels, classProbs]

    print "Using pre-computed superpixels and class label probabilities"
    if args.infile.endswith('.pkl'):
      superPixelInput = pomio.unpickleObject(args.infile)
      spix = superPixelInput[0]
      classProbs = superPixelInput[1]
    else:
      # a mat file, from paul S.
      matdata = scipy.io.loadmat( args.infile )['superpix']
      matlabels = matdata['label'][0,0] # labels start at 1 here
      matprobs  = matdata['prob'][0,0]
      # the labels are out of order for probabilities.  Paul has: impervious, bldg, car, low veg, tree, clutter
      matprobs = matprobs[:, np.array([1,2,4,5,3,6])-1]
      # currently my code relies on consecutive superpixels starting at 0
      ulabs = np.unique( matlabels )
      # replace matlabels values with renumbered labels.
      labelMap = np.zeros( (ulabs.max()+1,), dtype=int )
      for i,l in enumerate(ulabs):
        labelMap[l] = i
      matlabels = labelMap[ matlabels ]
      nodes, edges = superPixels.make_graph(matlabels) 
      spix = superPixels.SuperPixelGraph(matlabels,nodes,edges)
      # todo: I think I probably need to reduce these probs to correspond to the
      # ordinal labels in the superpixel graph.
      classProbs = matprobs[ ulabs-1, : ]
else:
    # Assume input image needs processing and classifying
    print "Superpixel generation mode"
    
    numberSuperPixels = args.nbSuperPixels
    superPixelCompactness = args.superPixelCompactness


    imgRGB = amntools.readImage( args.infile )

    # Turn image into superpixels.
    spix = superPixels.computeSuperPixelGraph( imgRGB, 'slic', [numberSuperPixels,superPixelCompactness] )

    print 'Loading classifier...'
    assert args.clfrFn != None, 'No classifier filename specified!'
        
    clfr = pomio.unpickleObject(args.clfrFn)

    print 'Computing superpixel features...'
    ftrs = features.computeSuperPixelFeatures( imgRGB, spix, ftype='classic', aggType='classic' )


    print 'Computing class probabilities...'
    classProbs = classification.classProbsOfFeatures(ftrs,clfr,\
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
        pomio.showLabels( spix.imageFromSuperPixelData( classLabs ) )
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

# Get adjacency probs
if args.adjFn != None:
  adjProbs = getAdjProbs(args.adjFn)
    
    
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

print '   done.'

colourMap = [\
    ('Impervious surfaces' , (255, 255, 255)),
    ('Building' ,            (0, 0, 255)),
    ('Low vegetation' ,      (0, 255, 255)),
    ('Tree' ,                (0, 255, 0)),
    ('Car' ,                 (255, 255, 0)),
    ('Clutter/background' ,  (255, 0, 0))\
]

if args.outfile and len(args.outfile)>0:
    print 'Writing output label file %s' % args.outfile
    outimg = pomio.msrc_convertLabelsToRGB( segResult, colourMap )
    skimage.io.imsave(args.outfile, outimg)
    print '   done.'

# Show the result.
if args.verbose:
    pomio.showLabels(segResult, colourMap)
    plt.title( 'Segmentation CRF result with K=%f' % K )
    plt.draw()
    print "labelling result, K = ", K 
    if dointeract:
        plt.waitforbuttonpress()



