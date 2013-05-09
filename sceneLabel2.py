#!/usr/bin/env python

# todo:
#   found this link too late:
#      http://opencvpython.blogspot.com/2013/03/histograms-4-back-projection.html
#   has much more succinct code

# todo: DONE convert this from cv to cv2.  See these links:
#    http://stackoverflow.com/questions/10417108/what-is-different-between-all-these-opencv-python-interfaces
#    
# OpenCV is dodgy as.  I couldn't use the histogram normalisation I wanted, 
#  and backproject doesn't work with 3D histograms.

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

cvimg = cv2.imread("ship-at-sea.jpg")
dimg = cvimg.copy()

dohsv = True

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
cv2.namedWindow("scenelabel2", 1)
cv2.imshow("scenelabel2", dimg)
print "Original image, press a key"
cv2.waitKey(0)
# now fg,bg probs
cv2.imshow("scenelabel2", pImgGivenFg)
print "foreground probs"
cv2.waitKey(0)
cv2.imshow("scenelabel2", pImgGivenBg)
print "background probs"
cv2.waitKey(0)

#img = imread("ship-at-sea.jpg")
#ppl.imshow(img)
#ppl.show()

sys.exit()

# Create the graph.
g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)
# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids, 50)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, img, 255-img)

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.
ppl.imshow(img2)
ppl.show()
