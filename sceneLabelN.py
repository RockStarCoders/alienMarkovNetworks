#!/usr/bin/env python

# SUMMARY: this is an N-class foreground/background labelling example:
#
# Usage:  ./sceneLabelN.py <classifierPkl> <imageName> 
#
# Example:
#
#     PYTHONPATH=./maxflow ./sceneLabelN.py randForestClassifier.pkl /home/jamie/data/MSRC_ObjCategImageDatabase_v2/Images/7_3_s.bmp

import pickle as pkl
import sys
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as plt
import scipy.ndimage.filters
import cython_uflow as uflow
import LRClassifier
import amntools
import sklearn
import sklearn.ensemble
import pomio
import FeatureGenerator

# parse args
clfrFn = sys.argv[1]
imgFn = sys.argv[2]

dointeract = 1
dbgMode = 0


#
# MAIN
#
imgRGB = imread( imgFn )
print 'Loading classifier...'
clfr = LRClassifier.loadClassifier(clfrFn)

print 'Computing class probabilities...'
classProbs = LRClassifier.generateImagePredictionClassDist(imgRGB, clfr,requireAllClasses=False)
print 'done.  result size = ', classProbs.shape

plt.interactive(1)
plt.imshow(imgRGB)
plt.title('original image')
#plt.waitforbuttonpress()

plt.figure()
print ' classes = ', clfr.classes_
# Transform class probs to the correct sized matrix.
nbRows = imgRGB.shape[0]
nbCols = imgRGB.shape[1]
nbClasses = pomio.getNumClasses()

cpnew = np.zeros( (nbRows, nbCols, nbClasses) )
for i in range( classProbs.shape[2] ):
    # stuff this set of probs to new label
    cpnew[:,:,clfr.classes_[i]] = classProbs[:,:,i] 
classProbs = cpnew
del cpnew

maxLabel = np.argmax( classProbs, 2 )

pomio.showLabels(maxLabel)
plt.title('raw clfr labels')

plt.figure()
pomio.showClassColours()

plt.draw()
if dointeract:
    plt.waitforbuttonpress()

print classProbs

if dbgMode:
    for i in range( classProbs.shape[2] ):
        plt.imshow( classProbs[:,:,i] )
        plt.title( 'class %d: %s' % (i,pomio.msrc_classLabels[i]) )
        plt.waitforbuttonpress()

nhoodSz = 4
sigsq = amntools.estimateNeighbourRMSPixelDiff(imgRGB,nhoodSz) ** 2
print "Estimated neighbour RMS pixel diff = ", np.sqrt(sigsq)

print "Performing maxflow for various smoothness K..."

# In Shotton, K0 and K in the edge potentials are selected manually from
# validation data results.
K0 = 0#0.5

plt.figure()
# for K in np.linspace(1,100,10):
for K in np.logspace(-2,1,5):
    # def nbrCallback( pixR, pixG, pixB, nbrR, nbrG, nbrB ):
    #    #print "*** Invoking callback"
    #    idiffsq = (pixR-nbrR)**2 + (pixG-nbrG)**2 + (pixB-nbrB)**2
    #    #idiffsq = (pixB-nbrB)**2
    #    res = np.exp( -idiffsq / (2 * sigsq) )
    #    #print res
    #    # According to Shotton, adding the constant can help remove
    #    # isolated pixels.
    #    res = K0 + res * K
    #    return res
    #
    # segResult = uflow.inferenceNCallback( \
    #     imgRGB.astype(float),\
    #     -np.log( np.maximum(1E-10, np.ascontiguousarray(classProbs) ) ), \
    #     'abswap',\
    #     nhoodSz, \
    #     nbrCallback )

    nbrPotentialMethod = 'contrastSensitive'
    nbrPotentialParams = [K0,K,sigsq]

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

