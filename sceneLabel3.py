#!/usr/bin/env python

"""
Command-line utility to do 3-class pixel-wise MRF segmentation.
"""

# SUMMARY: this is a 2-class foreground/background labelling example, really
# using a hidden MRF (obs not used in nbr potentials).  Fixed training
# rectangles are used to construct histograms, used for probability a pixel is
# foreground or background.  Grid weights encourage smoothness.  The
# segmentation is performed for various values of the smoothness parameter.
#
# The example uses the PyMaxflow library.  It exposes the main limitation, that
# the library can only accommodate one form of neighbourhood potential (unless
# I'm just not seeing it).  Would be better if we could supply a matrix with a
# probability table, to be more general.


# todo:
#   found this link too late:
#      http://opencvpython.blogspot.com/2013/03/histograms-4-back-projection.html
#   has much more succinct code

# todo: DONE convert this from cv to cv2.  See these links:
#    http://stackoverflow.com/questions/10417108/what-is-different-between-all-these-opencv-python-interfaces
#    
# OpenCV is dodgy as.  I couldn't use the histogram normalisation I wanted, 
#  and backproject doesn't work with 3D histograms.
#
# Edited the 2 class example to use Pymaxflow's alpha expansion

from matplotlib import pyplot as ppl
from maxflow import fastmin
from numpy.ma.core import exp
import cv2
import maxflow
import numpy as np
import scipy
import scipy.ndimage.filters
import sys
import random
import amntools

def extractHist( img, channels, ranges ):
    nc = img.shape[2]
    # controls resolution of histogram
    nbBins = len(channels) * [4]#[32, 32, 32]
    hist = cv2.calcHist( [img], channels, None, nbBins, ranges )
    return hist

def processHist( h ):
    # assuming it's float.  Make it sum to 1
    h = h / h.sum()
    assert feq( h.sum(), 1.0, 1E-5 ), "Histogram sums to %f" % h.sum()
    # now regularise by adding a uniform distribution with this much mass.
    regAlpha = 0.01
    h = regAlpha/h.size + (1.0-regAlpha)*h
    assert feq( h.sum(), 1.0, 1E-5 ), "Histogram sums to %f" % h.sum()
    return h

def feq(a,b,tol):
    return np.abs(a-b)<=tol

#
# MAIN
#
cvimg = amntools.readImage("ship-at-sea.jpg")
dimg = cvimg.copy()

dohsv = True
dointeract = False

if dohsv:
    # I would love to do this on rgb, but calcBackProject has a bug for
    # 3d histograms.  Flakey.
    # Convert to hsv instead
    cvimg = cv2.cvtColor(cvimg,cv2.COLOR_BGR2HSV)
    channels = [0,1]
    ranges = [0,180,0,256]
else:
    channels = [0,1,2]
    ranges = [0,256,0,256,0,256]

rectFg = (181,   196,    78,    35)
rectBg  = ( 10,    48,    55,    93)
# Draw them on a separate display image
cv2.rectangle( dimg, rectFg[0:2], \
                  (rectFg[0]+rectFg[2]-1, rectFg[1]+rectFg[3]-1), \
                  (0,0,255) ) # BGR, more opencv madness. it's for amateurs I think
cv2.rectangle( dimg, rectBg[0:2], \
                  (rectBg[0]+rectBg[2]-1, rectBg[1]+rectBg[3]-1), \
                  (0,0,255) ) 

z = cvimg[ rectFg[1]:rectFg[1]+rectFg[3], \
               rectFg[0]:rectFg[0]+rectFg[2] ]
histFg = extractHist( z, channels, ranges )
z = cvimg[ rectBg[1]:rectBg[1]+rectBg[3], \
               rectBg[0]:rectBg[0]+rectBg[2] ]
histBg = extractHist( z, channels, ranges )

cv2.imshow("scenelabel2-input", dimg)

# THIS IS WHAT SHOULD HAPPEN!
#
# Normalise and regularise histograms.  This will make sure the histograms
# contain no zeros, which will be useful when taking logs later.
# Note they are float32 numpy arrays
histFg = processHist( histFg )
histBg = processHist( histBg )

# BUT DUE TO OPENCV WEIRDNESS, JUST USE THE ONE EXAMPLE THAT WORKS.
#
# Turn into probability distributions
#cv2.normalize( histFg, histFg, 1.0, 0.0, cv2.NORM_L1, cv2.CV_64F )
# I couldn't get sum-to-1 normalisation working.  It's flakey this stuff
# So to treat these _like_ probabilities, divide by 255 first.
cv2.normalize( histFg, histFg, 0, 255, cv2.NORM_MINMAX)#, cv2.CV_64F )
cv2.normalize( histBg, histBg, 0, 255, cv2.NORM_MINMAX)#, cv2.CV_64F )


# print "Background histogram: ", histBg

# If you increase the resolution of the histogram you should blur it.
#sigma = 5
#histFg = 10*scipy.ndimage.filters.gaussian_filter(histFg,sigma)
#histBg = 100*scipy.ndimage.filters.gaussian_filter(histBg,sigma)

# Compute the probability of each pixel belonging to the fg/bg class given
# histograms (which we treat as probability distributions)
sf = 1.0
nc = cvimg.shape[2]
pImgGivenFg = cv2.calcBackProject( [cvimg], channels, histFg, \
                                       ranges, sf )

pImgGivenBg = cv2.calcBackProject( [cvimg], channels, histBg, \
                                       ranges, sf )

# # Display orig
# cv2.namedWindow("scenelabel2-input", 1)
# cv2.namedWindow("scenelabel2", 1)
# # moveWindow not in my version :(
# #cv2.moveWindow("scenelabel2-input",100,100)
# #cv2.moveWindow("scenelabel2",500,100)
# 
# cv2.imshow("scenelabel2-input", dimg)
# #print "Original image, press a key"
# #if dointeract:
# #    cv2.waitKey(0)
# # now fg,bg probs
# cv2.imshow("scenelabel3 Fg", pImgGivenFg)
# print "foreground probs"
# if dointeract:
#     cv2.waitKey(0)
# cv2.imshow("scenelabel3 Bg", pImgGivenBg)
# print "background probs"
# if dointeract:
#     cv2.waitKey(0)

# Looking in slides by S Gould, the interactive segmenation model (slide 18)
#
# http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=8&ved=0CG0QFjAH&url=http%3A%2F%2Fusers.cecs.anu.edu.au%2F~sgould%2Fpapers%2Fpart1-MLSS-2011.pdf&ei=pniKUZ6eMIS7igKPq4AI&usg=AFQjCNEzjnqqNyQY0TWbD1-pG57EI9jVgQ&sig2=RrmuwMX_TKj1ugK46mRxdg
#
# The unary term is the -log prob of fg/bg given obs, depending on source (bg)
# and sink (fg).  Note that our histograms are scaled to [0,255], and not
# properly normalised (thanks opencv).  Will do for the demo.
#
# The interactive terms are:
#
#     psi(yi,yj; x) = l0 + l1.exp(-||xi-xj||^2/2.beta)     if yi != yj
#                     0                                    otherwise
#
#
# We can't use PyMaxflow for this form of weights.  It seems it can only do:
# 
#    psi(yi,yj; x) = K.|yi-yj|
# 
# which doesn't use the image at all.  That's cool, this is just a demo.
   
   
# Alpha expansion for 2 class

numLabels = 3
imageSize = cvimg.shape[0:2]
xPixels = imageSize[0]
yPixels = imageSize[1]
numPixels = (xPixels * yPixels)

print "Image imageSize (1 channel) = ", imageSize, "total imageSize = " , numPixels


D = np.ndarray(shape=(numLabels, imageSize[0] * imageSize[1]))

# establish the unary costs in D, as per -log prob of fg/bg given obs
# cv2.imshow("background prior", pImgGivenBg)
# cv2.imshow("midground prior", (pImgGivenBg + pImgGivenFg) / 2)
# cv2.imshow("foreground prior", pImgGivenFg)
# cv2.waitKey(5000)

# colourNum = 255.0;
colourNum = 127.0;
colourNumInt = 127;

background = np.resize(-np.log(np.maximum(1E-10,pImgGivenBg.astype(float)/colourNum)),(numPixels))
midground = (np.resize(-np.log(np.maximum(1E-10,pImgGivenBg.astype(float)/colourNum)),(numPixels)) + np.resize(-np.log(np.maximum(1E-10,pImgGivenFg.astype(float)/255.0)), (numPixels))) / 2
midground = midground ** 2
foreground = np.resize(-np.log(np.maximum(1E-10,pImgGivenFg.astype(float)/255.0)), (numPixels))

for l in range(0, numLabels):
    if(l == 0):
        D[l] = background
    elif (l == 1):
        D[l] = midground / (max(midground))
    elif(l == 2):
        D[l] = foreground

maxCycles = 1000

for K in range(0,5):
    
    V = np.ndarray(shape=(numLabels, numLabels))
    print "size of pair cost array = ", np.shape(V)
    
    # establish the pair costs in V, as per psi(yi,yj; x) = l0 + l1.exp(-||xi-xj||^2/2.beta)     if yi != yj, 0 otherwise
    # class level neighbourhood costs
    for l1 in range(0, numLabels):
        for l2 in range(0, numLabels):
            # TODO can we implement pixel-specific neighbourhood costs?
            if l1 == l2:
                V[l1, l2] = 0
            else:
                V[l1, l2] = max(l1, l2) * K / (l1 + l2 + 1)
    
    print "Pairwise costs:\n" , V
    
    # set initial conditions
    labels = np.resize(0, (numPixels))
    
    for value in range(0,numPixels):
        a = random.random()
        if (0 < a < 0.33333):
            labels[value] = 0
            
        elif(0.33333 <= a < 0.66666):
            labels[value] = 1
            
        elif(a >= 0.66666):
            labels[value] = 2
    
    # initial labels
    initialConditionsImage = (np.resize(labels, (xPixels, yPixels))).astype('uint8')*colourNumInt;
    
    
    # perform fast approximate energy minimisation
    alphaExpansion = fastmin.aexpansion_grid(D, V, maxCycles, labels)
    alphaBetaSwap = fastmin.abswap_grid(D, V, maxCycles, labels)
    
    
    # Get the segment labels and convert to a format for image display
#     aeSegResultImage = (np.resize(alphaExpansion, (xPixels, yPixels))).astype('uint8')*colourNumInt
    
    aeSegResultImage = (np.resize(alphaExpansion.astype('uint8'), (xPixels, yPixels)))*colourNumInt
    
    print "Set of result image values::\n", set(np.resize(aeSegResultImage, numPixels))
    
    
    # normalise class labels for conversion to greyscale
    cv2.imshow("scenelabel3 initial labels", initialConditionsImage)
    cv2.imshow("scenelabel3 alpha expansion result", aeSegResultImage)
#     cv2.imshow("scenelabel3 alpha beta swap result", abSegResultImage)
    cv2.waitKey(4000)
