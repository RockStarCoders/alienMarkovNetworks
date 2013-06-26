#include <stdlib.h>
#include <exception>
#include <string>

typedef double (*NbrCallbackType)(
  double  pixR, double pixG, double pixB,
  double  nbrR, double nbrG, double nbrB,
  void*   cbdata
);

// Returns flow
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
  int32_t*        cMatOut
);


extern void ultraflow_inferenceN(
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
  int32_t*        cMatOut
);

class UflowException: public std::exception
{
  public:
    UflowException( const char* str ) throw() : m_str( str ) {}

    virtual ~UflowException() throw() {}

    virtual const char* what() const throw ()
    {
      return m_str.c_str();
    }

  protected:
    std::string m_str;
};

