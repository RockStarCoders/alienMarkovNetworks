#include "uflow.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <limits>
#include <boost/scoped_array.hpp>
#include <cmath>
#include <vector>

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
// Inline Functors for Neighbour Potentials
////////////////////////////////////////////////////////////////////////////////


class NbrPotentialFunctorCallback {
  public:
    NbrPotentialFunctorCallback(
      NbrCallbackType nbrEdgeCostCallback, void* nbrEdgeCostCallbackData
    )
      : m_callback( nbrEdgeCostCallback ), m_callbackData( nbrEdgeCostCallbackData )
    {
    }

    inline double operator()( 
      double R1, double G1, double B1, 
      double R2, double G2, double B2 
    ) const
    {
      // Params were never being used...
      return m_callback( R1, G1, B1, R2, G2, B2, m_callbackData );
    }

  private:
    NbrCallbackType m_callback;
    void* m_callbackData;
};

/////////////////////////////////////////////
template < typename T >
inline T sqr( const T& val ){ return val*val; }

class NbrPotentialFunctorContrastSensitive {
  public:
    NbrPotentialFunctorContrastSensitive( double K0, double K, double sigmaSq )
    :
      m_K0(K0), m_K(K), m_sigmaSq(sigmaSq)
    {
    }

    inline double operator()( 
      double R1, double G1, double B1, 
      double R2, double G2, double B2 
    ) const
    {
      const double idiffsq = sqr( R1-R2 ) + sqr( G1-G2 ) + sqr( B1-B2 );
      const double res = std::exp( -idiffsq / (2*m_sigmaSq) );

      // // an edge at a class boundary bears no penalty
      // const double res = ( R1>1E-10 || R2>1E-10 );

      return m_K0 + res*m_K;
    }

  private:
    const double m_K0;
    const double m_K;
    const double m_sigmaSq;
};

/////////////////////////////////////////////
class NbrPotentialFunctorDegreeSensitive {
  public:
    NbrPotentialFunctorDegreeSensitive(double K) : m_K(K) {}

    inline double operator()( double deg1, double deg2, int cl1, int cl2 )
    {
      return m_K / ( 0.5 * ( deg1 + deg2 ) );
    }
  private:
    const double m_K;
};

/////////////////////////////////////////////
class NbrPotentialFunctorAdjacencyAndDegreeSensitive {
  public:
    NbrPotentialFunctorAdjacencyAndDegreeSensitive(const double* adjProbs, int nbLabels, double K) 
      : m_adjProbs(adjProbs), m_nbLabels(nbLabels), m_K(K)
    {}

    inline double operator()( double deg1, double deg2, int cl1, int cl2 )
    {
      const double a = m_adjProbs[ cl1*m_nbLabels + cl2 ];
      return m_K * a / ( 0.5 * ( deg1 + deg2 ) );
    }
  private:
    const double* m_adjProbs;
    const int m_nbLabels;
    const double m_K;
};

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
static double inference2FunctorBased(
  int             nhoodSize,
  int             rows,
  int             cols,
  int             nbImgChannels,
  double*         cMatInputImage,
  double*         cMatSourceEdge,
  double*         cMatSinkEdge,
  FUNCTOR_TYPE&   functor,
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
            
            wt = functor(
              pixR, pixG, pixB,
              nbrR, nbrG,  nbrB
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
void computeSuperPixelDegree(
  int nbSuperPixels, int nbEdges, int32_t* cMatEdges, std::vector<int>& spDegree
)
{
  assert( spDegree.size() == nbSuperPixels );
  std::fill( spDegree.begin(), spDegree.end(), 0 );
  for ( int i=0; i<nbEdges; ++i )
  {
    const int32_t* ej = cMatEdges + 2*i;
    ++spDegree[ ej[0] ];
    ++spDegree[ ej[1] ];
  }

  // std::cout << "degree of superpixels:\n";
  // for ( int i=0; i<nbSuperPixels; ++i ){
  //   std::cout << "   " << i << " : " << spDegree[i] << "\n";
  // }
}

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
static double inference2SuperPixelFunctorBased(
  int             nbSuperPixels,
  int             nbEdges,
  int32_t*        cMatEdges,
  double*         cMatSourceEdge,
  double*         cMatSinkEdge,
  FUNCTOR_TYPE&   functor,
  int32_t*        cMatOut,
  bool*           validMask,
  int             srcClass,  // todo: hope I got this order right! doesn't matter.
  int             snkClass
)
{
  std::cout << "Inference superpixel 2-label:\n";

  // inference time  // The length nbSuperPixels array validMask indicates whether each pixel is a
  // valid part of the optimisation.  For example in ab-swaps non-ab pixels are
  // not part of it so have different nbr weights.  If the array is null then
  // not used at all.
  const bool dbg = false;
  const int n = nbSuperPixels;
  std::vector< int > spDegree( n );

  computeSuperPixelDegree( n, nbEdges, cMatEdges, spDegree );

  std::auto_ptr< GraphType > g(
    new GraphType(
      n,        /*estimated # of nodes*/
      nbEdges   /*estimated # of edges not inc src/snk*/
    )
  );

  int firstNode = g->add_node( n );
  assert( firstNode == 0 );

  // Add source and sink edge weights from given matrices.
  for ( int i=0; i<n; ++i ) {
    // std::cout << "sp " << i << ", adding src wt = " << cMatSourceEdge[i] 
    //           << ", snk wt = " <<  cMatSinkEdge[i] << "\n";
    g->add_tweights( i, cMatSourceEdge[i], cMatSinkEdge[i] );
  }

  // Now add the neighbour edges.
  for ( int i=0; i<nbEdges; ++i ){
    const int idx  = cMatEdges[2*i+0];
    const int nidx = cMatEdges[2*i+1];
    const bool validSP    = validMask==NULL || validMask[idx];
    const bool nbrValidSP = validMask==NULL || validMask[nidx];
    double wt;

    if ( validSP && nbrValidSP )
    {
      // todo: this is a bit shonky.  There is some adjacency probability for a
      // class to itself.  However in this case the weight will be zero.  Maybe
      // normalise probs relative to this prob, but how for difft classes?
      wt = functor( spDegree[idx], spDegree[nidx], srcClass, snkClass );
    }
    else
    {
      wt = 0.0;
    }
    // std::cout << "  edge (" << idx << "," << nidx << "), v=" << validSP
    //           << ", nv = " << nbrValidSP << ", wt = " << wt << "\n";
    g->add_edge( idx, nidx, wt, wt );
  }

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
  NbrPotentialFunctorCallback functor( nbrEdgeCostCallback, nbrEdgeCostCallbackData );
  return inference2FunctorBased(
    nhoodSize,
    rows,
    cols,
    nbImgChannels,
    cMatInputImage,
    cMatSourceEdge,
    cMatSinkEdge,
    functor,
    cMatOut,
    validMask
  );
}

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
double energyOfLabellingN(
  int             nhoodSize,
  int             rows,
  int             cols,
  int             nbImgChannels,
  int             nbLabels,
  double*         cMatInputImage,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
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
          res += functor(
            pixR, pixG, pixB,
            cMatInputImage[(idx+cols)*nbImgChannels+0],
            cMatInputImage[(idx+cols)*nbImgChannels+1],
            cMatInputImage[(idx+cols)*nbImgChannels+2] 
          );
        }
        // else Same label, no penalty.
      }
      if ( c < cols-1 )
      {
        // horizontal edge right of here
        if ( lbl != cMatLabels[idx+1] )
        {
          res += functor(
            pixR, pixG, pixB,
            cMatInputImage[(idx+1)*nbImgChannels+0],
            cMatInputImage[(idx+1)*nbImgChannels+1],
            cMatInputImage[(idx+1)*nbImgChannels+2] 
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
            res += functor(
              pixR, pixG, pixB,
              cMatInputImage[(idx+cols+1)*nbImgChannels+0],
              cMatInputImage[(idx+cols+1)*nbImgChannels+1],
              cMatInputImage[(idx+cols+1)*nbImgChannels+2]
            );
          }
          // else Same label, no penalty.
        }
        if ( r < rows-1 && c > 0 )
        {
          // diagonal edge down to left
          if ( lbl != cMatLabels[idx+cols-1] )
          {
            res += functor(
              pixR, pixG, pixB,
              cMatInputImage[(idx+cols-1)*nbImgChannels+0],
              cMatInputImage[(idx+cols-1)*nbImgChannels+1],
              cMatInputImage[(idx+cols-1)*nbImgChannels+2]
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
template < typename FUNCTOR_TYPE >
double energyOfLabellingNSuperPixel(
  int             nbSuperPixels,
  int             nbLabels,
  int             nbEdges,
  int32_t*        cMatEdges,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
  int32_t*        cMatLabels
)
{
  // Wot we got comin in:
  //
  //   cMatLabelWeights: -log P(x), that is they are potentials.
  //
  //   nbrEdgeCostCallback: -log P(xi,xj) given xi != xj
  double res = 0.0;

  std::vector< int > spDegree( nbSuperPixels );
  computeSuperPixelDegree( nbSuperPixels, nbEdges, cMatEdges, spDegree );

  // Sum of edge and node potentials.

  // Node potentials:
  for ( int i=0; i<nbSuperPixels; ++i ) {
    assert( 0 <= cMatLabels[i] && cMatLabels[i] < nbLabels );
    res += cMatLabelWeights[ i*nbLabels + cMatLabels[i] ];
  }
  std::cout << "dbg: just unary = " << std::fixed << res << "\n";

  // Nbr Edge potentials:
  // Only sum each edge once.
  for ( int i=0; i<nbEdges; ++i ) {
    const int32_t* ej = cMatEdges + 2*i;
    const int lbl = cMatLabels[ ej[0] ];
    if ( lbl != cMatLabels[ ej[1] ] )
    {
      const double ee = functor( spDegree[ ej[0] ], spDegree[ ej[1] ], lbl, cMatLabels[ ej[1] ] );
      // std::cout << "energy for edge " << ej[0] << ", " << ej[1] << " = "
      //           << ee << "\n";
      res += ee;
    }
    // else Same label, no penalty.
  }

  std::cout << "dbg: unary + binary = " << std::fixed << res << "\n";
  return res;
}

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
static void inferenceNABSwap(
  int             nhoodSize,
  int             rows,
  int             cols,
  int             nbImgChannels,
  int             nbLabels,
  double*         cMatInputImage,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
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

  // std::cout << "I think image ul pix = " << cMatInputImage[0]
  //           << ", " << cMatInputImage[1] << ", "
  //           << cMatInputImage[2] << "\n";

  // start with arbitrary labelling.  Note our current labelling is called "x"
  // in the alg (chaper 3 of the MRF book), here x == cMatOut.
  //
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
    functor,
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
    // for each UNIQUE pair of labels {a,b} in L
    for ( int a=0; a<nbLabels; ++a )
    {
      for ( int b=a+1; b<nbLabels; ++b )
      {
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
        double Exhat = inference2FunctorBased(
          nhoodSize,
          rows,
          cols,
          nbImgChannels,
          cMatInputImage,
          srcEdges.get(),
          snkEdges.get(),
          functor,
          t.get(),
          NULL
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
          functor,
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
// todo: repeated code...
template < typename FUNCTOR_TYPE >
static void inferenceSuperPixelABSwap(
  int             nbSuperPixels,
  int             nbLabels,
  int             nbEdges,
  int32_t*        cMatEdges,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
  int32_t*        cMatOut
)
{
  std::cout << "N-label AB swap algorithm, " << nbLabels << " labels.\n";
  
  // todo: parameterise
  const int maxIterations = 100;

  // start with arbitrary labelling.  Note our current labelling is called "x"
  // in the alg (chaper 3 of the MRF book), here x == cMatOut.
  std::fill( cMatOut, cMatOut + nbSuperPixels, 0 );
  // Compute energy of intial labelling.
  double Ex = energyOfLabellingNSuperPixel(
    nbSuperPixels,
    nbLabels,
    nbEdges,
    cMatEdges,
    cMatLabelWeights,
    functor,
    cMatOut
  );
  std::cout << "\t\t Initial energy = " 
            << std::fixed << std::setprecision(8)
            << Ex << "\n";

  bool success;
  boost::scoped_array< int32_t > t( new int32_t[nbSuperPixels] );
  boost::scoped_array< int32_t > proposedLabelling( new int32_t[nbSuperPixels] );
  boost::scoped_array< double > srcEdges( new double[nbSuperPixels] );
  boost::scoped_array< double > snkEdges( new double[nbSuperPixels] );
  boost::scoped_array< bool >   validMask( new bool[nbSuperPixels] );

  for ( int ic=0; ic<maxIterations; ++ic )
  {
    std::cout << "\t** iteration " << ic << "\n";

    success = false;
    // for each UNIQUE pair of labels {a,b} in L
    for ( int a=0; a<nbLabels; ++a )
    {
      for ( int b=a+1; b<nbLabels; ++b )
      {
        std::cout << "\t**  ab = " << a << "," << b << "\n";
        // find xhat = argmin E(x') among x' within one a-b swap of x

        // Use our 2-class inference to determine the transformation labels t.

        // Set up source and sink edges.
        for ( int i=0; i<nbSuperPixels; ++i )
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
        
        double Exhat = inference2SuperPixelFunctorBased(
          nbSuperPixels,
          nbEdges,
          cMatEdges,
          srcEdges.get(),
          snkEdges.get(),
          functor,
          t.get(),
          NULL, 
          a, 
          b
        );

        // To compute the energy, have to construct the proposed labelling.
        // We set x = xhat, which means use optimal move / transformation to
        // construct xhat.
        for ( int i=0; i<nbSuperPixels; ++i )
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

        Exhat = energyOfLabellingNSuperPixel(
          nbSuperPixels,
          nbLabels,
          nbEdges,
          cMatEdges,
          cMatLabelWeights,
          functor,
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
            proposedLabelling.get(),
            proposedLabelling.get()+nbSuperPixels,
            cMatOut
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
    std::cerr << "Warning: maximum iterations reached in inferenceSuperPixelABSwap"
              << std::endl;
  }
  std::cout << "** abswap complete!\n";
}

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
static void inferenceNUsingTFunctor(
  char*           method,
  int             nhoodSize,
  int             rows,
  int             cols,
  int             nbImgChannels,
  int             nbLabels,
  double*         cMatInputImage,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
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
      functor,
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

////////////////////////////////////////////////////////////////////////////////
template < typename FUNCTOR_TYPE >
static void inferenceSuperPixelUsingTFunctor(
  char*           method,
  int             nbSuperPixels,
  int             nbLabels,
  int             nbEdges,
  int32_t*        cMatEdges,
  double*         cMatLabelWeights,
  FUNCTOR_TYPE&   functor,
  int32_t*        cMatOut
)
{
  if ( method == std::string("abswap") )
  {
    inferenceSuperPixelABSwap(
      nbSuperPixels,
      nbLabels,
      nbEdges,
      cMatEdges,
      cMatLabelWeights,
      functor,
      cMatOut
    );
  }
  else if ( method == std::string("aexpansion") )
  {
    throw( UflowException( "alpha expansion not yet implemented" ) );
  }
  else
  {
    throw( UflowException( ("Unrecognised inferenceSuperPixel method '"
          + std::string(method) + "'").c_str() ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
// callback version
void ultraflow_inferenceNCallback(
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
  NbrPotentialFunctorCallback functor( nbrEdgeCostCallback, nbrEdgeCostCallbackData );
  inferenceNUsingTFunctor(
    method,
    nhoodSize,
    rows,
    cols,
    nbImgChannels,
    nbLabels,
    cMatInputImage,
    cMatLabelWeights,
    functor,
    cMatOut
  );
}

////////////////////////////////////////////////////////////////////////////////
// non-callback version
void ultraflow_inferenceN(
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
  int32_t*        cMatOut
)
{
  if ( nbrPotentialMethod == std::string("contrastSensitive") )
  {
    // shonky assignment
    double K0 = nbrPotentialParams[0];
    double K  = nbrPotentialParams[1];
    double sigmaSq = nbrPotentialParams[2];
    NbrPotentialFunctorContrastSensitive functor( K0, K, sigmaSq );
    inferenceNUsingTFunctor(
      method,
      nhoodSize,
      rows,
      cols,
      nbImgChannels,
      nbLabels,
      cMatInputImage,
      cMatLabelWeights,
      functor,
      cMatOut
    );
  }
  else
  {
    throw( UflowException( ("Unrecognised inferenceN nbr potential method '"
          + std::string(nbrPotentialMethod) + "'").c_str() ) );
  }
}
    
////////////////////////////////////////////////////////////////////////////////
// non-callback version
void ultraflow_inferenceSuperPixel(
  char*           method,
  int             nbSuperPixels,
  int             nbLabels,
  int             nbEdges,
  int32_t*        cMatEdges,
  double*         cMatLabelWeights,
  double*         cMatAdjProbs, // can be null
  char*           nbrPotentialMethod,
  double          K,
  int32_t*        cMatOut
)
{
  if ( nbrPotentialMethod == std::string("degreeSensitive") )
  {
    NbrPotentialFunctorDegreeSensitive functor(K);
    inferenceSuperPixelUsingTFunctor(
      method,
      nbSuperPixels,
      nbLabels,
      nbEdges,
      cMatEdges,
      cMatLabelWeights,
      functor,
      cMatOut
    );
  }
  else if ( nbrPotentialMethod == std::string("adjacencyAndDegreeSensitive") )
  {
    NbrPotentialFunctorAdjacencyAndDegreeSensitive functor(cMatAdjProbs, nbLabels, K);
    // todo: can get rid of this repeated code with a base class with virtual
    // oeprator method.  Because there's no polymorphism it will not incur the
    // cost of a virtual call.
    inferenceSuperPixelUsingTFunctor(
      method,
      nbSuperPixels,
      nbLabels,
      nbEdges,
      cMatEdges,
      cMatLabelWeights,
      functor,
      cMatOut
    );
  }
  else
  {
    throw( UflowException( (
          "Unrecognised inferenceSuperPixel nbr potential method '"
          + std::string(nbrPotentialMethod) + "'").c_str() ) );
  }
}

