// Functions that use boost max flow to perform inference for image labelling.
// Rather use kolmogorov's code, it's proven and easier to use.


// http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays

// C extension module to numpy
#include "Python.h"
#include "numpy/arrayobject.h"
//#include "numpy/ndarraytypes.h"

#include "C_uflow.h"
//#include "graph.h"

static PyMethodDef _C_uflowMethods[] = { 
  {"uflow_inference2", uflow_inference2, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

void init_C_uflow() { 
  (void) Py_InitModule("_C_uflow", _C_uflowMethods);
  import_array(); // Must be present for NumPy. Called first after above line.
}



/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
/* template < typename T > */
/* T *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  { */
/*     int i,n; */
    
/*     n=arrayin->dimensions[0]; */
/*     return (T *) arrayin->data;  /\* pointer to arrayin data as double *\/ */
/* } */


/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublevector: array must be of type Float and 1 dimensional (n).");
        return 1;  }
    return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
    if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
        return 1;  }
    return 0;
}



/****************************/

void ultraflow_inference2( int nhoodSize, int rows, int cols, double* cMatSourceEdge, double* cMatSinkEdge, double* cMatInputImage, 
  const char* nbrEdgeCostMethod, double* cCallbackParams, int* cMatOut )
{
  /* do nothing for now */
}

/****************************/
static PyObject *uflow_inference2(PyObject *self, PyObject *args)
{
  /* 
   * In python you call it like this:
   *
   *                                        matrix-2d     matrix-2d     matrix-3d    integer    string            array-1d
   *    labelResult* uflow_inference2( sourceEdgeCosts, sinkEdgeCosts, inputImage, nhoodSize, nbrEdgeCostMethod, callbackParams )
   *
   * where callback fn has this signature:
   *
   *    float edgeCallback( floatImage3DMat, pixelRowInt, pixelColInt, pixelNbrRowInt, pixelNbrColInt, floatParams1DArray )
   */
  PyArrayObject *matSourceEdge, *matSinkEdge, *matInputImage, *arrCallbackParams,    *matOut;
  double *cMatSourceEdge, *cMatSinkEdge, *cCallbackParams, *cMatInputImage;
  int *cMatOut;
  // callback?
  int nhoodSize;
  int rows, cols, dims[2];/* Parse tuples separately since args will differ between C fcns */
  const char* nbrEdgeCostMethod;

  if (!PyArg_ParseTuple(args, "O!O!O!isO!",
      &PyArray_Type, &matSourceEdge, &PyArray_Type, &matSinkEdge, &PyArray_Type, &matInputImage, &nhoodSize, &nbrEdgeCostMethod, &PyArray_Type, &arrCallbackParams)) return NULL;

  if (NULL == matSourceEdge || NULL==matSinkEdge || NULL==matInputImage || NULL==arrCallbackParams) return NULL; 


  /* Check that object input is 'double' type and a matrix
     Not needed if python wrapper function checks before call to this routine */
  if (not_doublematrix(matSourceEdge)) return NULL;
  if (not_doublematrix(matSinkEdge)) return NULL;
  if (not_doublematrix(matInputImage)) return NULL;
  if (not_doublevector( arrCallbackParams)) return NULL;

  /* Get the dimensions of the input */
  rows=dims[0]=matSourceEdge->dimensions[0];
  cols=dims[1]=matSourceEdge->dimensions[1];
  assert( rows == matSinkEdge->dimensions[0] && cols == matSinkEdge->dimensions[1] );
  assert( rows == matInputImage->dimensions[0] && cols == matInputImage->dimensions[1] );

  /* /\* turn arrays into c pointers, linear, but row major *\/ */
  /* cMatSourceEdge = pyvector_to_Carrayptrs<double>( matSourceEdge ); */
  /* cMatSinkEdge   = pyvector_to_Carrayptrs<double>( matSinkEdge ); */
  /* cMatInputImage = pyvector_to_Carrayptrs<double>( matSourceEdge ); */
  /* cCallbackParams = pyvector_to_Carrayptrs<double>( arrCallbackParams ); */
  
  /* /\* Make a new integer matrix of same dims *\/ */
  /* matOut = (PyArrayObject *) PyArray_FromDims(2,dims,NPY_INT); */
  /* cMatOut   = pyvector_to_Carrayptrs<int>( matOut ); */

  /* computation... */
  ultraflow_inference2( nhoodSize, rows, cols, cMatSourceEdge, cMatSinkEdge, cMatInputImage, nbrEdgeCostMethod, cCallbackParams, 
    cMatOut );

  return PyArray_Return(matOut); 
}



