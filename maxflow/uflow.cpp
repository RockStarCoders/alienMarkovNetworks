#include "uflow.hpp"

#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <boost/scoped_array.hpp>

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
double ultraflow_inference2(
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
)
{
  const int n = rows*cols;  
  assert( nbImgChannels == 3 ); // currently only support RGB
  assert( nhoodSize == 4 || nhoodSize == 8 );
  const int (*nhood)[2] = ( nhoodSize == 4 ) ? s_nhood4 : s_nhood8;
  const int nhoodLen    = ( nhoodSize == 4 ) ? 2        : 4;

  std::cout << "ultraflow_inference2: nhoodSize = " << nhoodSize
            << ", img size = (" << rows << ", " << cols << ", " 
            << nbImgChannels << ")"
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
      // Assume planar rgb storage
      double pixR = cMatInputImage[idx],
        pixG = cMatInputImage[idx + n],
        pixB = cMatInputImage[idx + 2*n];

      for ( int j=0; j<nhoodLen; ++j ){
        const int nr = r + nhood[j][0];
        const int nc = c + nhood[j][1];
        if ( 0 <= nr && nr < rows && 0 <= nc && nc < cols )
        {
          const int nidx = nr*cols + nc;
          double nbrR = cMatInputImage[nidx],
            nbrG = cMatInputImage[nidx + n],
            nbrB = cMatInputImage[nidx + 2*n];

          const double wt = nbrEdgeCostCallback(
            pixR, pixG, pixB,
            nbrR, nbrG,  nbrB,
            nbrEdgeCostCallbackData 
          );
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

  return flow;
}


////////////////////////////////////////////////////////////////////////////////
void inferenceNABSwap(
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
)
{
  // todo: for now include whole graph.  But actually the graph only needs to
  // have nodes whose label is alpha or beta.  See PyMaxFlow for example.  It's
  // more complicated because then the cut weight is not the energy of the
  // labelling.
  
  const int maxIterations = 100;
  const int npix = rows*cols;

  // start with arbitrary labelling.  Note our current labelling is called "x"
  // in the alg (chaper 3 of the MRF book), here x == cMatOut.
  std::fill( cMatOut, cMatOut + npix, 0 );
  // Rather than complicated computation of the energy of this labelling, assume
  // it's not the optimal result and so the next result will be better.
  // todo: fix?
  double Ex = std::numeric_limits<double>::infinity();
  bool success;
  boost::scoped_array< int32_t > t( new int32_t[npix] );
  boost::scoped_array< double > srcEdges( new double[npix] );
  boost::scoped_array< double > snkEdges( new double[npix] );

  for ( int ic=0; ic<maxIterations; ++ic )
  {
    success = false;
    // for each pair of labels {a,b} in L
    for ( int a=0; a<nbLabels; ++a )
    {
      for ( int b=0; b<nbLabels; ++b )
      {
        // find xhat = argmin E(x') among x' within one a-b swap of x

        // Use our 2-class inference to determine the transformation labels t.

        // Set up source and sink edges.
        for ( int i=0; i<npix; ++i )
        {
          assert( 0 <= cMatOut[i] && cMatOut[i] < nbLabels );

          if ( cMatOut[i] == a || cMatOut[i] == b )
          {
            // t == 0 case, label a is assigned.
            srcEdges[i] = cMatLabelWeights[ i + npix*a ];
            // t == 1 case, label b is assigned.
            snkEdges[i] = cMatLabelWeights[ i + npix*b ];
          }
          else
          {
            // This pixel is not a candidate for swapping.  Arbitrarily
            // associate with the source.  Guarantee that by setting one
            // weight to Inf.
            srcEdges[i] = cMatLabelWeights[ i + npix*cMatOut[i] ];
            snkEdges[i] = std::numeric_limits<double>::infinity();
          }
        }
        
        // For the edge weight we need to wrap the simple 2-class callback 
        // with a function of the 
        // todo: make nbr cost as function of a-b class labels
        const double Exhat = ultraflow_inference2(
          nhoodSize,
          rows,
          cols,
          nbImgChannels,
          cMatInputImage,
          srcEdges.get(),
          snkEdges.get(),
          nbrEdgeCostCallback,
          nbrEdgeCostCallbackData,
          t.get()
        );
       
        // If E(xhat) < E(x) set x = xhat and success = 1
        if ( Exhat < Ex )
        {
          // We set x = xhat, which means use optimal move / transformation to
          // construct xhat.
          for ( int i=0; i<npix; ++i )
          {
            if ( cMatOut[i] == a || cMatOut[i] == b )
            {
              // A candidate for swap.  Depends on t.
              cMatOut[i] = t[i] ? b : a;
            }
          }
          Ex = Exhat;
          success = true;
        }

      }// for b
    }// for a

    if ( !success )
    {
      break;
    }
  }// for ic

  if ( success )
  {
    std::cerr << "Warning: maximum iterations reached in inferenceNABSwap"
              << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
void ultraflow_inferenceN(
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
)
{
  if ( method == std::string("abswap") )
  {
    inferenceNABSwap(
      nhoodSize,
      rows,
      cols,
      nbImgChannels,
      nbLabels,
      cMatInputImage,
      cMatLabelWeights,
      nbrEdgeCostCallback,
      nbrEdgeCostCallbackData,
      cMatOut
    );
  }
  else if ( method == std::string("aexpansion") )
  {
    throw( UflowException( "alpha expansion not yet implemented" ) );
  }
  else
  {
    throw( UflowException( ("Unrecognised inferenceN method '"
          + std::string(method) + "'").c_str() ) );
  }
}
