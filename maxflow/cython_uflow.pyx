import cython

# callback example:
#
#   https://github.com/cython/cython/tree/master/Demos/callback
#

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "uflow.hpp": # essential!
    ctypedef double (*NbrCallbackType)(
        double  pixR, double pixG, double pixB,
        double  nbrR, double nbrG, double nbrB,
        void*   cbdata
        )
        
cdef extern from "uflow.hpp": # essential!
    extern void ultraflow_inference2(
      int             nhoodSize,
      int             rows,
      int             cols,
      int             nbImgChannels,
      double*         cMatInputImage,
      double*         cMatSourceEdge,
      double*         cMatSinkEdge,
      NbrCallbackType nbrEdgeCostCallback,
      void*           nbrEdgeCostCallbackData,
      np.int32_t*     cMatOut
    )


#  cdef void ultraflow_inference2( 
#    int nhoodSize, int rows, int cols, double* cMatSourceEdge, 
#    double* cMatSinkEdge, double* cMatInputImage, 
#    char* nbrEdgeCostMethod, 
#    double (*nbrCallback)(void *), void* pycallback,
#    double* cCallbackParams, np.int32_t* cMatOut )
#
cdef double callbackWrapper(   
    double  pixR, double pixG, double pixB,
    double  nbrR, double nbrG, double nbrB,
    void *f ):
    return (<object>f)( pixR, pixG, pixB, nbrR, nbrG, nbrB )

# The purpose of this cython module is to make calls to uflow functions
# available from python.  So we have to turn numpy nd arrays into c pointers
# to pass to uflow functions.
#
# to build:
#
#   > cd maxflow
#   > python setup.py build_ext --inplace


def inference2( np.ndarray[double, ndim=3, mode="c"] inputImage not None,
                np.ndarray[double, ndim=2, mode="c"] sourceEdgeCosts not None,
                np.ndarray[double, ndim=2, mode="c"] sinkEdgeCosts not None,
                int nhoodSize,
                nbrEdgeCallback ):
    rows = sourceEdgeCosts.shape[0]
    cols = sourceEdgeCosts.shape[1]

    assert( nhoodSize == 4 or nhoodSize == 8 )
    assert( sourceEdgeCosts.shape == sinkEdgeCosts.shape, \
                "edge costs not same size: src = %s, snk = %s" % \
                (str(sourceEdgeCosts), str(sinkEdgeCosts)) )
    assert( inputImage.ndim == 3 and inputImage.shape[0] == rows and 
            inputImage.shape[1] == cols )

    imgChannels = inputImage.shape[2]

    #  create output label array
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] labelResult = \
        np.zeros( (rows,cols), dtype=np.int32 )

    # Call C++ inference function
    ultraflow_inference2( nhoodSize, rows, cols, imgChannels,
                          &inputImage[0,0,0], 
                          &sourceEdgeCosts[0,0], 
                          &sinkEdgeCosts[0,0], 
                          callbackWrapper,
                          <void*>nbrEdgeCallback,
                          &labelResult[0,0] )

    return labelResult
