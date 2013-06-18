import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "uflow.hpp": # essential!
  cdef void ultraflow_inference2( 
    int nhoodSize, int rows, int cols, double* cMatSourceEdge, 
    double* cMatSinkEdge, double* cMatInputImage, 
    char* nbrEdgeCostMethod, double* cCallbackParams, np.int32_t* cMatOut )


# The purpose of this cython module is to make calls to uflow functions
# available from python.  So we have to turn numpy nd arrays into c pointers
# to pass to uflow functions.

# 
# In python you call it like this:
# 
#                                        matrix-2d     matrix-2d     matrix-3d    integer    string            array-1d
#    labelResult* uflow_inference2( sourceEdgeCosts, sinkEdgeCosts, inputImage, nhoodSize, nbrEdgeCostMethod, callbackParams )
# 
# where callback fn has this signature:
# 
#    float edgeCallback( floatImage3DMat, pixelRowInt, pixelColInt, pixelNbrRowInt, pixelNbrColInt, floatParams1DArray )
# 
def inference2( np.ndarray[double, ndim=2, mode="c"] sourceEdgeCosts not None,
                np.ndarray[double, ndim=2, mode="c"] sinkEdgeCosts not None,
                np.ndarray[double, ndim=3, mode="c"] inputImage not None,
                int nhoodSize,
                char* nbrEdgeCostMethod, 
                np.ndarray[double, ndim=1, mode="c"] callbackParams not None ):
    
    rows = sourceEdgeCosts.shape[0]
    cols = sourceEdgeCosts.shape[1]

    assert( sourceEdgeCosts.shape == sinkEdgeCosts.shape, \
                "edge costs not same size: src = %s, snk = %s" % \
                (str(sourceEdgeCosts), str(sinkEdgeCosts)) )
    assert( inputImage.ndim == 3 and inputImage.shape[0] == rows and 
            inputImage.shape[1] == cols )
    assert( nhoodSize == 4 or nhoodSize == 8 )
    assert( callbackParams.ndim == 1 )

    #  create output label array
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] labelResult = np.zeros( (rows,cols), dtype=np.int32 )

    # Call C++ inference function
    ultraflow_inference2( nhoodSize, rows, cols, &sourceEdgeCosts[0,0], 
                          &sinkEdgeCosts[0,0], &inputImage[0,0,0], 
                          nbrEdgeCostMethod, &callbackParams[0], 
                          &labelResult[0,0] )

    return labelResult
