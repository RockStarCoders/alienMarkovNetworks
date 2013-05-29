#!/usr/bin/env python

# SUMMARY: this is a simple N-class labelling example, without higher
# order cliques.
#
# The example uses the PyMaxflow library. In the previous example I was wrong,
# it can handle multi-label crfs.

import sys
import cv2
import maxflow
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl
import scipy.ndimage.filters

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
cvimg = cv2.imread("ship-at-sea.jpg")
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


print "Background histogram: ", histBg

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

# Display orig
cv2.namedWindow("scenelabel2-input", 1)
cv2.namedWindow("scenelabel2", 1)
# moveWindow not in my version :(
#cv2.moveWindow("scenelabel2-input",100,100)
#cv2.moveWindow("scenelabel2",500,100)

cv2.imshow("scenelabel2-input", dimg)
#print "Original image, press a key"
#if dointeract:
#    cv2.waitKey(0)
# now fg,bg probs
cv2.imshow("scenelabel2", pImgGivenFg)
print "foreground probs"
if dointeract:
    cv2.waitKey(0)
cv2.imshow("scenelabel2", pImgGivenBg)
print "background probs"
if dointeract:
    cv2.waitKey(0)

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

print "Performing maxflow for various smoothness K..."

# for K in np.linspace(1,100,10):
for K in np.logspace(0,3,10):
   # Create the graph.  Float capacities.
   g = maxflow.Graph[float]()
   # Add the nodes. nodeids has the identifiers of the nodes in the grid.
   # Same x-y extent as the image, but just 1 channel/band.
   nodeids = g.add_grid_nodes(cvimg.shape[0:2])
   # Add non-terminal edges with the same capacity.
   g.add_grid_edges(nodeids, K)
   # Add the terminal edges. The image pixels are the capacities
   # of the edges from the source node. The inverted image pixels
   # are the capacities of the edges to the sink node.
   # Don't let log arg drop to zero.
   g.add_grid_tedges(nodeids, \
                         -np.log(np.maximum(1E-10,pImgGivenFg.astype(float)/255.0)),\
                         -np.log(np.maximum(1E-10,pImgGivenBg.astype(float)/255.0)) )
   
   # Find the maximum flow.
   g.maxflow()
   # Get the segments of the nodes in the grid.  A boolean array that is true
   # where foreground (sink), and false where background (source).
   segResult = g.get_grid_segments(nodeids)
   
   # Show the result.
   cv2.imshow("scenelabel2", segResult.astype('uint8')*255)
   print "segmentation result, K = ", K 
   cv2.waitKey(500)
