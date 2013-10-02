import cython

# callback example:
#
#   https://github.com/cython/cython/tree/master/Demos/callback
#

# The purpose of this cython module is to make calls to uflow functions
# available from python.  So we have to turn numpy nd arrays into c pointers
# to pass to uflow functions.
#
# to build:
#
#   > cd maxflow
#   > python setup.py build_ext --inplace



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
    extern double ultraflow_inference2(
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

cdef extern from "uflow.hpp": # essential!
    extern void ultraflow_inferenceNCallback(
      char*           method,
      int             nhoodSize,
      int             rows,
      int             cols,
      int             nbImgChannels,
      int             nbLabels,
      double*         cMatInputImage,
      double*         cMatLabelWeights,
      NbrCallbackType nbrEdgeCostCallback,
      void*           nbrEdgeCostCallbackData,
      np.int32_t*     cMatOut
    )

cdef extern from "uflow.hpp": # essential!
    extern void ultraflow_inferenceN(
      char*           method,
      int             nhoodSize,
      int             rows,
      int             cols,
      int             nbImgChannels,
      int             nbLabels,
      double*         cMatInputImage,
      double*         cMatLabelWeights,
      char*           nbrPotentialMethod, 
      double*         nbrPotentialParams,
      np.int32_t*     cMatOut
    )

cdef extern from "uflow.hpp": # essential!
    extern void ultraflow_inferenceSuperPixel(
      char*           method,
      int             nbSuperPixels,
      int             nbLabels,
      int             nbEdges,
      np.int32_t*     cMatEdges,
      double*         cMatLabelWeights,
      double*         cMatAdjProbs, # can be null
      char*           nbrPotentialMethod, 
      double          K,
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

    # make sure arrays are c contiguous
    assert inputImage.flags['C_CONTIGUOUS']
    assert sourceEdgeCosts.flags['C_CONTIGUOUS']
    assert sinkEdgeCosts.flags['C_CONTIGUOUS']

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



# 'method' can be aexpansion or abswap
def inferenceN( np.ndarray[double, ndim=3, mode="c"] inputImage not None,
                np.ndarray[double, ndim=3, mode="c"] labelWeights not None,
                method,
                int nhoodSize,
                nbrPotentialMethod,
                np.ndarray[double, ndim=1, mode="c"] nbrPotentialParams not None ):
    rows = labelWeights.shape[0]
    cols = labelWeights.shape[1]

    assert( method == 'abswap' or method == 'aexpansion' )
    assert( nhoodSize == 4 or nhoodSize == 8 )
    print 'Input image has shape %s, should be (%d,%d,3)' \
        % (str(np.shape(inputImage)),rows,cols)
    assert inputImage.ndim == 3 and inputImage.shape[0] == rows and \
        inputImage.shape[1] == cols

    nbLabels = labelWeights.shape[2]
    assert( nbLabels > 1, "Only 1 label class?" );

    # make sure contiguous
    assert inputImage.flags['C_CONTIGUOUS']
    assert labelWeights.flags['C_CONTIGUOUS']
    assert nbrPotentialParams.flags['C_CONTIGUOUS']

    imgChannels = inputImage.shape[2]

    #  create output label array
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] labelResult = \
        np.zeros( (rows,cols), dtype=np.int32 )

    # Call C++ inference function
    ultraflow_inferenceN( method, nhoodSize, rows, cols, imgChannels, nbLabels,
                          &inputImage[0,0,0], 
                          &labelWeights[0,0,0], 
                          nbrPotentialMethod, 
                          &nbrPotentialParams[0],
                          &labelResult[0,0] )

    return labelResult

# 'method' can be aexpansion or abswap
def inferenceNCallback( np.ndarray[double, ndim=3, mode="c"] inputImage not None,
                np.ndarray[double, ndim=3, mode="c"] labelWeights not None,
                method,
                int nhoodSize,
                nbrEdgeCallback ):
    rows = labelWeights.shape[0]
    cols = labelWeights.shape[1]

    #print "***!!! img ul = %f,%f,%f" % (inputImage[0,0,0], inputImage[0,0,1], inputImage[0,0,2] )

    assert( method == 'abswap' or method == 'aexpansion' )
    assert( nhoodSize == 4 or nhoodSize == 8 )
    print 'Input image has shape %s, should be (%d,%d,3)' \
        % (str(np.shape(inputImage)),rows,cols)
    assert inputImage.ndim == 3 and inputImage.shape[0] == rows and inputImage.shape[1] == cols

    nbLabels = labelWeights.shape[2]
    assert( nbLabels > 1, "Only 1 label class?" );

    # make sure contiguous
    assert inputImage.flags['C_CONTIGUOUS']
    assert labelWeights.flags['C_CONTIGUOUS']

    imgChannels = inputImage.shape[2]

    #  create output label array
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] labelResult = \
        np.zeros( (rows,cols), dtype=np.int32 )

    # Call C++ inference function
    ultraflow_inferenceNCallback( method, nhoodSize, rows, cols, imgChannels, nbLabels,
                          &inputImage[0,0,0], 
                          &labelWeights[0,0,0], 
                          callbackWrapper,
                          <void*>nbrEdgeCallback,
                          &labelResult[0,0] )

    return labelResult



# 'method' can be aexpansion or abswap
def inferenceSuperPixel(\
    superPixelGraph,
    np.ndarray[double, ndim=2, mode="c"] labelWeights not None,
    np.ndarray[double, ndim=2, mode="c"] adjProbs, # can be None
    method,
    nbrPotentialMethod,
    K ):
    #    np.ndarray[double, ndim=1, mode="c"] nbrPotentialParams not None ):

    N = superPixelGraph.getNumSuperPixels()
    assert( labelWeights.ndim == 2 and labelWeights.shape[0] == N );
    assert( method == 'abswap' or method == 'aexpansion' )

    nbLabels = labelWeights.shape[1]
    assert( nbLabels > 1, "Only 1 label class?" );

    # make sure contiguous
    assert labelWeights.flags['C_CONTIGUOUS']
    #assert nbrPotentialParams.flags['C_CONTIGUOUS']

    # turn edge array into an Nx2 matrix
    # cython no like
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] edgeMat = \
        np.vstack( superPixelGraph.m_edges ).astype(np.int32)
    #     cdef np.ndarray[np.int32_t, ndim=2, mode="c"] edgeMat
    #     for i,e in enumerate(superPixelGraph.m_edges):
    #         edgeMat[i,0] = e[0]
    #         edgeMat[i,1] = e[1]
    print type(edgeMat)
    print edgeMat
    assert edgeMat.shape[1] == 2 

    # create output label array of length nbSuperPixels
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] labelResult = \
        np.zeros( (N), dtype=np.int32 )

    # Call C++ inference function.  This will get us a label per superpixel.
    cdef double* adjProbsRef
    if adjProbs == None:
        adjProbsRef = NULL
    else:
        adjProbsRef = &adjProbs[0,0]

    ultraflow_inferenceSuperPixel(
        method,
        N,
        nbLabels,
        len( superPixelGraph.m_edges ), 
        &edgeMat[0,0],
        &labelWeights[0,0], 
        adjProbsRef,
        nbrPotentialMethod, 
        K,
        &labelResult[0] 
      )
        #&nbrPotentialParams[0],

    # turn labels to image array
#    print 'LabelResult has shape ', np.shape(labelResult), ' and is ', \
#        labelResult
    return superPixelGraph.imageFromSuperPixelData( \
        np.reshape(labelResult, (len(labelResult),1) ) )
