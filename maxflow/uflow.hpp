#include <stdlib.h>

typedef double (*NbrCallbackType)(
  double  pixR, double pixG, double pixB,
  double  nbrR, double nbrG, double nbrB,
  void*   cbdata
);

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
  int32_t*        cMatOut
);


