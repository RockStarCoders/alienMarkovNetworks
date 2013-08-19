#!/usr/bin/env python
import numpy as np
import cython_uflow as uflow
import SuperPixels as sp

# Test functions in cython_uflow.pyx



#
# Super Pixel inference
#

# Construct test data: a 2x2 checkerboard with 4 superpixels.  Want top 2
# to come out class 0, and bottom to be class 1.
splabels = np.array( [
        [0,0,0, 1,1,1],
        [0,0,0, 1,1,1],
        [0,0,0, 1,1,1],
        [2,2,2, 3,3,3],
        [2,2,2, 3,3,3],
        [2,2,2, 3,3,3] ], dtype=np.int32 )
spnodes = np.arange(4)
spedges = [ [0,1], [2,3], [0,2], [1,3] ];
lblWts = -np.log( np.array( [ [ 0.01, 0.02, 0.90], 
                   [ 0.80, 0.02, 0.05], 
                   [ 0.01, 0.02, 0.87], 
                   [ 0.01, 0.82, 0.03] ] ) )

spgraph = sp.SuperPixelGraph(splabels,spnodes,spedges)

x = np.reshape( [np.arange(4)*10], (4,1) )
y = spgraph.imageFromSuperPixelData( x )
print y

res = uflow.inferenceSuperPixel( spgraph, lblWts, 'abswap', 'degreeSensitive' )
print "Inference result = \n", res

expectedResult = np.array( [
        [2,2,2, 0,0,0],
        [2,2,2, 0,0,0],
        [2,2,2, 0,0,0],
        [2,2,2, 1,1,1],
        [2,2,2, 1,1,1],
        [2,2,2, 1,1,1] ] )
assert( np.all( expectedResult == res ) )
