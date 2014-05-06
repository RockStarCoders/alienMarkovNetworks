"""
Second cut library to:
   * generate features from image per pixel
   * generate features per super-pixel (by aggregation)
Has some more generality than FeatureGenerator.py

The default data storage is examples x dimensions, fortran style, because this
is what matlab does and has ended up what sklearn etc does.
"""

import superPixels
import FeatureGenerator
import numpy as np
import scipy
import multiprocessing as mp

# Returns a DxN matrix, D the feature dimension and N the number of pixels.
def computePixelFeatures( rgbImage, ftype ):
  res = []

  if ftype == 'classic':
    res = FeatureGenerator.generatePixelFeaturesForImage( rgbImage )
  else:
    raise Exception('Invalid feature type "%s"' % ftype)

  assert res.shape[1] > 0
  assert res.shape[0] == rgbImage.shape[0]*rgbImage.shape[1]
  assert np.all( np.isfinite( res ) )
  return res

# Returns a DxN matrix, D is the aggregate feature dimension and N is the number of super pixels.
def aggregateFeaturesBySuperPixel( pixelFeatures, superPixelsObj, aggtype ):
  res = []
  Np, Dp = pixelFeatures.shape
  N = superPixelsObj.getNumSuperPixels()

  if aggtype == 'classic':
    # Same as FeatureGenerator.py generateSuperPixelFeatures
    dim = 0
    D = 4*Dp + 1
    res = np.zeros( (N,D), dtype=float )
    # Turn label image into same dim as matrix width
    labs = superPixelsObj.getLabelImage().flatten()
    assert len(labs) == Np
    assert N-1 == labs.max() and 0 == labs.min()
    # Visit each super pixel
    for i in range(N):
      X = pixelFeatures[labs==i,:]
      assert X.shape[0] > 0, "Empty superpixel!"
      with np.errstate( invalid='ignore' ):
        res[i,:] = np.concatenate([
            X.mean(dim),
            X.std(dim),
            scipy.stats.skew(X,dim),
            scipy.stats.kurtosis(X,dim),
            [X.shape[dim]]
            ])
  else:
    raise Exception('Invalid super-pixel feature aggregation type "%s"' % aggtype)

  assert res.shape[1] > 0
  assert res.shape[0] == N
  assert np.all( np.isfinite( res ) )
  return res


def computeSuperPixelFeatures( rgbImage, superPixelsObj, ftype, aggtype ):
  pixelFeatures = computePixelFeatures( rgbImage, ftype )
  spFeatures = aggregateFeaturesBySuperPixel(
    pixelFeatures, superPixelsObj, aggtype
    )
  return spFeatures

def computeSuperPixelFeaturesMulti(
  images, superPixelObjs, ftype, aggtype, asMatrix, nbCores=1
  ):
  assert len(images) == len(superPixelObjs)
  if nbCores>1:
    job_server = mp.Pool(nbCores)
    jres = [ job_server.apply_async( computeSuperPixelFeatures, \
                                       ( img, spo, ftype, aggtype ) ) \
               for img,spo in zip(images, superPixelObjs) ]
    tOutSecs = 10*60 # 10 mins
    res = [ jr.get(timeout=tOutSecs) for jr in jres ]
  else:
    res = [ computeSuperPixelFeatures( img, spo, ftype, aggtype ) for \
              img,spo in zip(images, superPixelObjs) ]
  if asMatrix:
    return np.vstack( res )
  else:
    return res
