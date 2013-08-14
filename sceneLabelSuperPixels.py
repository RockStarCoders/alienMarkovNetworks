#!/usr/bin/env python

# Usage:  ./sceneLbaelSuperPixels.py <classifierPkl> <imageName> 
#
# Example:
#
#     PYTHONPATH=./maxflow ./sceneLabelSuperPixels.py randForestClassifierSP.pkl /home/jamie/data/MSRC_ObjCategImageDatabase_v2/Images/7_3_s.bmp

import pickle as pkl
import sys
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as plt
import scipy.ndimage.filters
import cython_uflow as uflow
import sklearn
import sklearn.ensemble
import bonzaClassifier
import amntools
import pomio
#import FeatureGenerator
import slic
import SuperPixels

# parse args
clfrFn = sys.argv[1]
imgFn = sys.argv[2]

dointeract = 1
dbgMode = 0


#
# MAIN
#
imgRGB = imread( imgFn )

# Turn image into superpixels.
spix = SuperPixels.computeSuperPixelGraph( imgRGB, 'slic', [100,10] )

print 'Loading classifier...'
clfr = LRClassifier.loadClassifier(clfrFn)

print 'Computing superpixel features...'
ftrs = churchillFeatures.computeSuperPixelFeatures( imgRGB, spix )

print 'Computing class probabilities...'
classProbs = bonzaClassifier.classpProbsOfFeatures(ftrs,clfr,\
                                                       requireAllClasses=False)


plt.interactive(1)
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

# Do the inference
# prefer to merge regions with high degree
nbrPotentialMethod = 'degreeSensitive'
nbrPotentialParams = [K0,K,sigsq]
print 'Performing CRF inference...'
segResult = uflow.inferenceN( \
    imgRGB.astype(float),\
    -np.log( np.maximum(1E-10, np.ascontiguousarray(classProbs) ) ), \
    'abswap',\
    nhoodSz, \
    nbrPotentialMethod, np.ascontiguousarray(nbrPotentialParams) )

# Show the result.
pomio.showLabels(segResult)
plt.title( 'Segmentation with K=%f' % K )
plt.draw()
print "labelling result, K = ", K 
if dointeract:
    plt.waitforbuttonpress()

#plt.interactive(False)
#plt.show()

