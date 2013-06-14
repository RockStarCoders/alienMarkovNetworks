// Functions that use boost max flow to perform inference for image labelling.
// Rather use kolmogorov's code, it's proven and easier to use.

// C extension module to numpy
#include "Python.h"
#include "arrayobject.h"

static PyObject *uflow_inference2(PyObject *self, PyObject *args);
