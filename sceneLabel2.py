#!/usr/bin/env python

# todo:
#   found this link too late:
#      http://opencvpython.blogspot.com/2013/03/histograms-4-back-projection.html
#   has much more succinct code

# todo: convert this from cv to cv2.  See these links:
#    http://stackoverflow.com/questions/10417108/what-is-different-between-all-these-opencv-python-interfaces
#    
import sys
import cv
import maxflow
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl

def imgPlanesAsImgArray( img ):
    # Split the image out into the RGB channels
    R = cv.CreateMat( img.rows, img.cols, cv.CV_8UC1 )
    G = cv.CreateMat( img.rows, img.cols, cv.CV_8UC1 )
    B = cv.CreateMat( img.rows, img.cols, cv.CV_8UC1 )
    return [cv.GetImage(z) for z in [R, G, B]]

def extractHist( img ):
    binSz = [32, 32, 32]
    rngs = [ [0, 255], [0, 255], [0, 255] ]
    hist = cv.CreateHist( binSz, cv.CV_HIST_ARRAY, rngs, 1 )
    cv.CalcHist( imgPlanesAsImgArray(img), hist )
    return hist

cvimg = cv.LoadImageM("ship-at-sea.jpg")
cv.NamedWindow("Fg/Bg Seg Example", 1)

rectFg = (181,   196,    78,    35)
rectBg  = ( 10,    48,    55,    93)
# Draw them on a separate display image
dimg = cv.CloneMat(cvimg)
cv.Rectangle( dimg, rectFg[0:2], \
                  (rectFg[0]+rectFg[2]-1, rectFg[1]+rectFg[3]-1), \
                  cv.RGB(255,0,0) )
cv.Rectangle( dimg, rectBg[0:2], \
                  (rectBg[0]+rectBg[2]-1, rectBg[1]+rectBg[3]-1), \
                  cv.RGB(255,0,0) ) 

histFg = extractHist( cv.GetSubRect(cvimg,rectFg) )
histBg = extractHist( cv.GetSubRect(cvimg,rectBg) )
# Turn into probability distributions
cv.NormalizeHist( histFg, 1.0 )
cv.NormalizeHist( histBg, 1.0 )

# Compute the probability of each pixel belonging to the fg/bg class given
# histograms (which we treat as probability distributions)
pImgGivenFg = cv.CreateMat( cvimg.rows, cvimg.cols, cv.CV_8UC1 )
pImgGivenBg = cv.CreateMat( cvimg.rows, cvimg.cols, cv.CV_8UC1 )
cv.CalcBackProject( imgPlanesAsImgArray( cvimg ), pImgGivenFg, histFg )
cv.CalcBackProject( imgPlanesAsImgArray( cvimg ), pImgGivenBg, histBg )

# Display orig
cv.ShowImage("Input Image", dimg)
cv.WaitKey(0)
# now fg,bg probs
cv.ShowImage("Foreground probs", pImgGivenFg)
cv.WaitKey(0)
cv.ShowImage("Background probs", pImgGivenBg)
cv.WaitKey(0)

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
