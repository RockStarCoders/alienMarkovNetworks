#include "uflow.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <limits>
#include <boost/scoped_array.hpp>

#include "graph.h"

// NOTE: numpy arrays are stored so that the last dimension changes fastest.
//  for example, a 3-d matrix RxCxD would be indexed by: x[d + c*D + r*(C*D)]

// todo: nclass need to make edge callback return 0 for non a-b edges
//   Need to explicitly compute energy for whole graph.

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
  int32_t*        cMatOut,
  bool*           validMask
)
{
  // The wxh array validMask indicates whether each pixel is a valid part of the
  // optimisation.  For example in ab-swaps non-ab pixels are not part of it so
  // have different nbr weights.  If the array is null then not used at all.
  const int n = rows*cols;  
  assert( nbImgChannels == 3 ); // currently only support RGB
  assert( nhoodSize == 4 || nhoodSize == 8 );
  const int (*nhood)[2] = ( nhoodSize == 4 ) ? s_nhood4 : s_nhood8;
  const int nhoodLen    = ( nhoodSize == 4 ) ? 2        : 4;

  const bool dbg = false;

  if (dbg)
  {
    std::cout << "ultraflow_inference2: nhoodSize = " << nhoodSize
              << ", img size = (" << rows << ", " << cols << ", " 
              << nbImgChannels << ")"
              << std::endl;

    std::cout << "nhood offsets:" << std::endl;
    for ( int i=0; i<nhoodLen; ++i ){
      std::cout << "  " << nhood[i][0] << ", " << nhood[i][1] << std::endl;
    }
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
      double pixR = cMatInputImage[idx*nbImgChannels+0],
        pixG = cMatInputImage[idx*nbImgChannels+1],
        pixB = cMatInputImage[idx*nbImgChannels+2];

      bool validPix = validMask==NULL || validMask[idx];

      for ( int j=0; j<nhoodLen; ++j ){
        const int nr = r + nhood[j][0];
        const int nc = c + nhood[j][1];
        if ( 0 <= nr && nr < rows && 0 <= nc && nc < cols )
        {
          const int nidx = nr*cols + nc;
          bool nbrValidPix = validMask==NULL || validMask[nidx];
          double wt;

          if ( validPix && nbrValidPix )
          {
            double nbrR = cMatInputImage[nidx*nbImgChannels+0],
              nbrG = cMatInputImage[nidx*nbImgChannels+1],
              nbrB = cMatInputImage[nidx*nbImgChannels+2];
            
            wt = nbrEdgeCostCallback(
              pixR, pixG, pixB,
              nbrR, nbrG,  nbrB,
              nbrEdgeCostCallbackData 
            );
          }
          else
          {
            wt = 0.0;
          }
          g->add_edge( idx, nidx, wt, wt );
          ++edgeCt;
        }
      }// for j
    }// for c
  }// for r

  assert( edgeCt == nbNEdges );

  // inference time
  double flow = g->maxflow();

  if (dbg)
  {
    std::cout << "Flow = " << flow << std::endl;
  }

  // Store min cut labels in output array.
  for ( int i=0; i<n; ++i )
  {
    cMatOut[i] = g->what_segment( i );
  }

  return flow;
}


////////////////////////////////////////////////////////////////////////////////
double energyOfLabellingN(
  int             nhoodSize,
  int             rows,
  int             cols,
  int             nbImgChannels,
  int             nbLabels,
  double*         cMatInputImage,
  double*         cMatLabelWeights,
  NbrCallbackType nbrEdgeCostCallback,
  void*           nbrEdgeCostCallbackData,
  int32_t*        cMatLabels
)
{
  // Wot we got comin in:
  //
  //   cMatLabelWeights: -log P(x), that is they are potentials.
  //
  //   nbrEdgeCostCallback: -log P(xi,xj) given xi != xj
  double res = 0.0;
  const int npix = rows*cols;

  // Sum of edge and node potentials.

  // Node potentials:
  for ( int i=0; i<npix; ++i ) {
    assert( 0 <= cMatLabels[i] && cMatLabels[i] < nbLabels );
    res += cMatLabelWeights[ i*nbLabels + cMatLabels[i] ];
  }
  std::cout << "dbg: just unary = " << std::fixed << res << "\n";
  const double un = res;

  // Nbr Edge potentials:
  // Only sum each edge once.
  
  // vertical and horizontal
  int idx = 0;
  for ( int r=0; r<rows; ++r ) {
    for ( int c=0; c<cols; ++c, ++idx ) {
      const int lbl = cMatLabels[idx];
      double pixR = cMatInputImage[idx*nbImgChannels+0],
        pixG = cMatInputImage[idx*nbImgChannels+1],
        pixB = cMatInputImage[idx*nbImgChannels+2];

      if ( r < rows-1 )
      {
        // vertical edge down from here
        if ( lbl != cMatLabels[idx+cols] )
        {
          res += nbrEdgeCostCallback(
            pixR, pixG, pixB,
            cMatInputImage[(idx+cols)*nbImgChannels+0],
            cMatInputImage[(idx+cols)*nbImgChannels+1],
            cMatInputImage[(idx+cols)*nbImgChannels+2],
            nbrEdgeCostCallbackData 
          );
        }
        // else Same label, no penalty.
      }
      if ( c < cols-1 )
      {
        // horizontal edge right of here
        if ( lbl != cMatLabels[idx+1] )
        {
          res += nbrEdgeCostCallback(
            pixR, pixG, pixB,
            cMatInputImage[(idx+1)*nbImgChannels+0],
            cMatInputImage[(idx+1)*nbImgChannels+1],
            cMatInputImage[(idx+1)*nbImgChannels+2],
            nbrEdgeCostCallbackData 
          );
        }
        // else Same label, no penalty.
      }
    }// for c
  }// for r
  std::cout << "dbg: just binary = " << std::fixed << res-un << "\n";

  if ( nhoodSize == 8 ) {
    // Add two sets of diagonal edges too.
    int idx = 0;
    for ( int r=0; r<rows; ++r ) {
      for ( int c=0; c<cols; ++c, ++idx ) {
        const int lbl = cMatLabels[idx];
        double pixR = cMatInputImage[idx*nbImgChannels+0],
          pixG = cMatInputImage[idx*nbImgChannels+1],
          pixB = cMatInputImage[idx*nbImgChannels+2];

        if ( r < rows-1 && c < cols - 1 )
        {
          // diagonal down to right
          if ( lbl != cMatLabels[idx+cols+1] )
          {
            res += nbrEdgeCostCallback(
              pixR, pixG, pixB,
              cMatInputImage[(idx+cols+1)*nbImgChannels+0],
              cMatInputImage[(idx+cols+1)*nbImgChannels+1],
              cMatInputImage[(idx+cols+1)*nbImgChannels+2],
              nbrEdgeCostCallbackData 
            );
          }
          // else Same label, no penalty.
        }
        if ( r < rows-1 && c > 0 )
        {
          // diagonal edge down to left
          if ( lbl != cMatLabels[idx+cols-1] )
          {
            res += nbrEdgeCostCallback(
              pixR, pixG, pixB,
              cMatInputImage[(idx+cols-1)*nbImgChannels+0],
              cMatInputImage[(idx+cols-1)*nbImgChannels+1],
              cMatInputImage[(idx+cols-1)*nbImgChannels+2],
              nbrEdgeCostCallbackData 
            );
          }
          // else Same label, no penalty.
        }
      }// for c
    }// for r
  }

  return res;
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
  std::cout << "N-label AB swap algorithm, " << nbLabels << " labels.\n";
  // todo: for now include whole graph.  But actually the graph only needs to
  // have nodes whose label is alpha or beta.  See PyMaxFlow for example.  It's
  // more complicated because then the cut weight is not the energy of the
  // labelling.
  
  const int maxIterations = 100;
  const int npix = rows*cols;

  std::cout << "I think image ul pix = " << cMatInputImage[0]
            << ", " << cMatInputImage[1] << ", "
            << cMatInputImage[2] << "\n";

  // start with arbitrary labelling.  Note our current labelling is called "x"
  // in the alg (chaper 3 of the MRF book), here x == cMatOut.
  //
  // Initialising to a constant doesn't work.  xi==xj for all neighbours, so no
  // nbr potentials ever get added.  It's a degenerate global minimum!
  std::fill( cMatOut, cMatOut + npix, 0 );
  // Compute energy of intial labelling.
  double Ex = energyOfLabellingN(
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
  std::cout << "\t\t Initial energy = " 
            << std::fixed << std::setprecision(8)
            << Ex << "\n";

  bool success;
  boost::scoped_array< int32_t > t( new int32_t[npix] );
  boost::scoped_array< int32_t > proposedLabelling( new int32_t[npix] );
  boost::scoped_array< double > srcEdges( new double[npix] );
  boost::scoped_array< double > snkEdges( new double[npix] );
  boost::scoped_array< bool >   validMask( new bool[npix] );

  for ( int ic=0; ic<maxIterations; ++ic )
  {
    std::cout << "\t** iteration " << ic << "\n";

    success = false;
    // for each pair of labels {a,b} in L
    for ( int a=0; a<nbLabels; ++a )
    {
      for ( int b=0; b<nbLabels; ++b )
      {
        if (a==b) continue;
        std::cout << "\t**  ab = " << a << "," << b << "\n";
        // find xhat = argmin E(x') among x' within one a-b swap of x

        // Use our 2-class inference to determine the transformation labels t.

        // Set up source and sink edges.
        for ( int i=0; i<npix; ++i )
        {
          assert( 0 <= cMatOut[i] && cMatOut[i] < nbLabels );

          if ( cMatOut[i] == a || cMatOut[i] == b )
          {
            // t == 0 case, label a is assigned.  Cut snk edge with higher prob,
            // lower potential ==> put alpha weight on snk edge.
            snkEdges[i] = cMatLabelWeights[ i*nbLabels + a ];
            // t == 1 case, label b is assigned.
            srcEdges[i] = cMatLabelWeights[ i*nbLabels + b ];
            validMask[i] = true;
          }
          else
          {
            // This pixel is not a candidate for swapping.  Arbitrarily
            // associate with the source.  Guarantee that by setting one
            // weight to Inf.
            snkEdges[i] = cMatLabelWeights[ i*nbLabels + cMatOut[i] ];
            srcEdges[i] = std::numeric_limits<double>::infinity();
            validMask[i] = false;
          }
        }// for i
        
        // For the edge weight we need to wrap the simple 2-class callback 
        // with a function of the 
        // todo: make nbr cost as function of a-b class labels
        double Exhat = ultraflow_inference2(
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

        // To compute the energy, have to construct the proposed labelling.
        // We set x = xhat, which means use optimal move / transformation to
        // construct xhat.
        for ( int i=0; i<npix; ++i )
        {
          if ( cMatOut[i] == a || cMatOut[i] == b )
          {
            // A candidate for swap.  Depends on t.
            proposedLabelling[i] = t[i] ? b : a;
          }
          else
          {
            proposedLabelling[i] = cMatOut[i];
          }
        }

        Exhat = energyOfLabellingN(
          nhoodSize,
          rows,
          cols,
          nbImgChannels,
          nbLabels,
          cMatInputImage,
          cMatLabelWeights,
          nbrEdgeCostCallback,
          nbrEdgeCostCallbackData,
          proposedLabelling.get()
        );
        std::cout << "\t\t Computed energy = " 
                  << std::fixed << std::setprecision(8)
                  << Exhat << "\n";

        // If E(xhat) < E(x) set x = xhat and success = 1
        if ( Exhat < Ex )
        {
          Ex = Exhat;
          std::cout << "\t**  went downhill, criterion Ex = " << Ex << "\n";
          std::copy(
            proposedLabelling.get(), proposedLabelling.get()+npix, cMatOut
          );
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
  std::cout << "** abswap complete!\n";
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
