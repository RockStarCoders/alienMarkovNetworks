#!/usr/bin/env python

import sys
import cv
import maxflow
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl

cvimg = cv.LoadImageM("ship-at-sea.jpg")
cv.NamedWindow("Input image", 1)

rectShip = (181,   196,    78,    35)
rectSea  = ( 10,    48,    55,    93)
# Draw them
dimg = cv.CloneMat(cvimg)
cv.Rectangle( dimg, rectShip[0:2], \
                  (rectShip[0]+rectShip[2]-1, rectShip[1]+rectShip[3]-1), \
                  cv.RGB(255,0,0) )
cv.Rectangle( dimg, rectSea[0:2], \
                  (rectSea[0]+rectSea[2]-1, rectSea[1]+rectSea[3]-1), \
                  cv.RGB(255,0,0) ) 

cv.ShowImage("Input Image", dimg)
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
