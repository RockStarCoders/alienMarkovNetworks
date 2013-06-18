#include <iostream>
#include <stdlib.h>
#include <memory>

#include "graph.h"

// Type used in computations.
typedef double DType;
typedef Graph<DType,DType,DType> GraphType;

const int s_nhood4[][2] = { // r, c
  {0,1}, // h-edge to right
  {1,0}  // v-edge to bottom
};

const int s_nhood8[][2] = { // r, c
  {0,1}, // h-edge to right
  {1,0}, // v-edge to bottom
  {1,1}, // d-edge bottom-right
  {1,-1},// d-edge bottom-left
};

////////////////////////////////////////////////////////////////////////////////
void ultraflow_inference2(
  int         nhoodSize,
  int         rows,
  int         cols,
  double*     cMatSourceEdge,
  double*     cMatSinkEdge,
  double*     cMatInputImage,
  const char* nbrEdgeCostMethod,
  double*     cCallbackParams,
  int32_t*    cMatOut
)
{
  const double K = cCallbackParams[0]; // !!
  const int n = rows*cols;  
  assert( nhoodSize == 4 || nhoodSize == 8 );
  const int (*nhood)[2] = ( nhoodSize == 4 ) ? s_nhood4 : s_nhood8;
  const int nhoodLen    = ( nhoodSize == 4 ) ? 2        : 4;

  std::cout << "ultraflow_inference2: nhoodSize = " << nhoodSize
            << ", img size = (" << rows << ", " << cols
            << "), edge cost method = " << nbrEdgeCostMethod
            << ", K = " << K
            << std::endl;

  std::cout << "nhood offsets:" << std::endl;
  for ( int i=0; i<nhoodLen; ++i ){
    std::cout << "  " << nhood[i][0] << ", " << nhood[i][1] << std::endl;
  }

  const int nbNEdges = ( nhoodSize==4 ) ? (rows-1)*cols + (cols-1)*rows 
    : (rows-1)*cols + (cols-1)*rows + 2*(rows-1)*(cols-1);

  std::auto_ptr< GraphType > g(
    new GraphType(
      n,        /*estimated # of nodes*/
      nbNEdges  /*estimated # of edges not inc src/snk*/
    )
  );

  int firstNode = g->add_node( n );
  assert( firstNode == 0 );

  // Add source and sink edge weights from given matrices.
  for ( int i=0; i<n; ++i ) {
    g->add_tweights( i, cMatSourceEdge[i], cMatSinkEdge[i] );
  }

  // Now add the neighbour edges.
  // Careful not to add edges twice!
  // The method is: 
  //    for each pixel, add a horizontal edge to right (4,8-conn)
  //    for each pixel, add a vertical   edge to bottom (4,8-conn)
  //    for each pixel, add a diagonal   edge to bottom-right (8-conn)
  //    for each pixel, add a diagonal   edge to bottom-left (8-conn)
  // and if the other end is not in the image, don't add the edge.
  int idx = 0; // this pixel index
  int edgeCt = 0; // for asserting
  for ( int r=0; r<rows; ++r ){
    for ( int c=0; c<cols; ++c, ++idx ){
      for ( int j=0; j<nhoodLen; ++j ){
        const int nr = r + nhood[j][0];
        const int nc = c + nhood[j][1];
        if ( 0 <= nr && nr < rows && 0 <= nc && nc < cols )
        {
          const int nidx = nr*cols + nc;
          const double wt = K;//edgeWtFunctor(
          //            cMatInputImage, rows, cols, imgDims, r, c, nr, nc
          //);
          g->add_edge( idx, nidx, wt, wt );
          ++edgeCt;
        }
      }// for j
    }// for c
  }// for r

  assert( edgeCt == nbNEdges );

  // inference time
	double flow = g->maxflow();

  std::cout << "Flow = " << flow << std::endl;

  // Store min cut labels in output array.
  for ( int i=0; i<n; ++i )
  {
    cMatOut[i] = g->what_segment( i );
  }
}
