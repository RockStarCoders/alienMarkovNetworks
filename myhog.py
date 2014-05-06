# /usr/local/lib/python2.7/dist-packages/skimage/feature/_hog.py

import pdb 
import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
import pedocc

def checkrange( x, vmin, vmax ):
  assert np.all( np.logical_and( vmin <= x, x <= vmax ) )

def getmax( X, Y ):
  assert X.shape == Y.shape
  # Get the value where the absolute value is maximum.
  # There is a general way to do this for 3D matrices using np.indices,
  # but this is easier to understand.
  inds = np.argmax( np.abs( np.dstack( [ X, Y ] ) ), axis=2 )
  res = X.copy()
  msk = (inds==1)
  res[msk] = Y[msk]
  return res

# Can handle RGB, takes max
def computeGradients( image, normalise ):
  if image.ndim == 3:
    # colour.  take maximum gradient
    gx, gy = computeGradients( image[:,:,0], normalise )
    # Slightly tricky because we want to select the gradient whose absolute
    # value is maximum.
    gxnew, gynew = computeGradients( image[:,:,1], normalise )
    gx = getmax( gx, gxnew )
    gy = getmax( gy, gynew )
    del gxnew, gynew
    gxnew, gynew = computeGradients( image[:,:,2], normalise )
    gx = getmax( gx, gxnew )
    gy = getmax( gy, gynew )
    del gxnew, gynew
  else:
    # luminance
    assert image.ndim == 2

    origtype = image.dtype
    if image.dtype.kind == 'u':
      # convert uint image to float
      # to avoid problems with subtracting unsigned numbers in np.diff()
      image = image.astype(pedocc.ftype)
    else:
      image = image.copy()

    # JRS: Assuming our images are on the scale 0-255, convert to [0,1]
    if origtype.itemsize == 1:
      image /= 255.0
    elif origtype.itemsize == 2:
      image /= 65535.0
    else:
      assert False, 'Unexpected data depth %d bytes for type %s' \
          % (origtype.itemsize,str(origtype))

    checkrange( image, 0.0, 1.0 )

    if normalise:
        image = sqrt(image)

    zc = np.zeros((image.shape[0],1))
    zr = np.zeros((1,image.shape[1]))
    gx = np.hstack( [ zc, image[:,2:] - image[:,:-2], zc ] )
    gy = np.vstack( [ zr, image[2:,:] - image[:-2,:], zr ] )
    assert gx.shape == image.shape
    assert gy.shape == image.shape
    assert np.abs(gx).max() <= 1.0 and np.abs(gy).max() <= 1.0, \
        'Max grad x = %f, y = %f' % (np.abs(gx).max(),np.abs(gy).max())

  # in either case return result
  assert gx.ndim == 2 and gx.shape == gy.shape and gx.shape == image.shape[:2]
  return gx, gy

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False,
        gx=None, gy=None, flatten=True):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the HOG.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    image = np.atleast_2d(image)

    """
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    #if image.ndim > 2:
    #    raise ValueError("Currently only supports grey-level images")
    assert image.ndim <= 3

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if gx == None:
      assert gy == None
      gx, gy = computeGradients( image, normalise )

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180
    # release some memory
    del gx, gy

    sy, sx = image.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    histdims = (n_cellsy, n_cellsx, orientations)
    histnumel = np.prod( histdims )
    if 0:
      # JRS: ORIGINAL IMPLEMENTATION
      orientation_histogram = np.zeros( histdims )
      subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]
      for i in range(orientations):
          #create new integral image for this orientation
          # isolate orientations in this range

          temp_ori = np.where(orientation < 180.0 / orientations * (i + 1),
                              orientation, -1)
          temp_ori = np.where(orientation >= 180.0 / orientations * i,
                              temp_ori, -1)
          # select magnitudes for those orientations
          cond2 = temp_ori > -1
          temp_mag = np.where(cond2, magnitude, 0)

          # JRS: error in original, the uniform filter sums to 1
          temp_filt = uniform_filter(temp_mag, size=(cy, cx)) * cx * cy
          orientation_histogram[:, :, i] = temp_filt[subsample]
    else:
      # JRS: trilinear interpolation
      # Find lower and upper histogram corners for each pixel.
      # For each pixel, work out the indices and weights contributed to the 8 
      # neighbouring bins. This comes from page 118 of Dalal's thesis.
      #
      # Note the image dims might not be divisible by the cell size.  In this
      # case We trim the lower-right border off of orientation and magnitude
      # before further processing.
      sxtrunc = n_cellsx*cx
      sytrunc = n_cellsy*cy
      xpix, ypix = np.meshgrid( range(sxtrunc), range(sytrunc), copy=True )
      obinsz = 180.0 / orientations
      # Turn into bin co-ords
      x = xpix.astype(pedocc.ftype).flatten() / float( cx )
      y = ypix.astype(pedocc.ftype).flatten() / float( cy )
      orient = orientation[:sytrunc,:sxtrunc].flatten() / obinsz
      # rem because some orientations can be exactly on 180
      orient = orient % orientations
      mag = magnitude[:sytrunc,:sxtrunc].flatten()
      x0 = x.astype(np.int32)
      y0 = y.astype(np.int32)
      orient0 = orient.astype(np.int32)
      x1 = x0 + 1
      y1 = y0 + 1
      # to deal with angle wrap around
      orient1 = ( orient0 + 1 ) % orientations
      assert np.all( orient1 < orientations )
      dx = x - x0
      del x
      dy = y - y0
      del y
      dorient = orient - orient0
      del orient

      checkrange( dx, 0.0, 1.0 )
      checkrange( dy, 0.0, 1.0 )
      checkrange( dorient, 0.0, 1.0 )

      # For each of the 8 neighbouring bins add a contribution for each pixel.
      # Note now all the bin sizes are 1 because we scaled back to those coords
      contribx, contriby, contriborient, contribwt = [], [], [], []

      # Can go out of bounds to the lower right for x1,y1 (not orient1 because
      # of wrap around).  Simple case is to clamp to number of bins.  Might be
      # better in future to not use this contribution since it might create
      # artefacts at the boundary.
      #
      # Later note: yes, there are artefacts:
      #
      #   + on the upper-left edge the HOG counts are dimmer, because the x0-x1
      #   relation always looks toward south-east.
      #
      #   + on lower-right edge, out-of-bounds contributions are added to the
      #   last cell, making it brighter.
      #
      # Does it matter?  Only if what is being compared during training and test
      # differ.  At the moment, this is the case because during training, the
      # image bounds are the same as the hog array extent.  Those artefacts
      # occur at the edges of the training examples.  During testing (scanning)
      # the array is much larger and where we compute the HOG array there are no
      # edge artefacts.
      #
      # Two ways to ameliorate this:
      #
      #   1) during training, compute hog features for larger window and crop out centre of features.
      #
      #   2) dump the lower-right edge OOB components, at least then the dimness
      #   is consistent with upper-left edge.  Could later double these
      #   artificially.

      checkrange( x0, 0, histdims[1]-1 )
      checkrange( y0, 0, histdims[0]-1 )
      checkrange( x1, 0, histdims[1] )
      checkrange( y1, 0, histdims[0] )
      checkrange( orient0, 0, histdims[2]-1 )
      checkrange( orient1, 0, histdims[2]-1 )

      # x0,y0,orient0
      contribx.append( x0 )
      contriby.append( y0 )
      contriborient.append( orient0 )
      contribwt.append( mag * (1.0 - dx) * (1.0 - dy) * (1.0 - dorient) )

      # x0,y0,orient1
      contribx.append( x0 )
      contriby.append( y0 )
      contriborient.append( orient1 )
      contribwt.append( mag * (1.0 - dx) * (1.0 - dy) * (      dorient) )

      # x0,y1,orient0
      contribx.append( x0 )
      contriby.append( y1 )
      contriborient.append( orient0 )
      contribwt.append( mag * (1.0 - dx) * (      dy) * (1.0 - dorient) )

      # x1,y0,orient0
      contribx.append( x1 )
      contriby.append( y0 )
      contriborient.append( orient0 )
      contribwt.append( mag * (      dx) * (1.0 - dy) * (1.0 - dorient) )

      # x0,y1,orient1
      contribx.append( x0 )
      contriby.append( y1 )
      contriborient.append( orient1 )
      contribwt.append( mag * (1.0 - dx) * (      dy) * (      dorient) )

      # x1,y0,orient1
      contribx.append( x1 )
      contriby.append( y0 )
      contriborient.append( orient1 )
      contribwt.append( mag * (      dx) * (1.0 - dy) * (      dorient) )

      # x1,y1,orient0
      contribx.append( x1 )
      contriby.append( y1 )
      contriborient.append( orient0 )
      contribwt.append( mag * (      dx) * (      dy) * (1.0 - dorient) )

      # x1,y1,orient1
      contribx.append( x1 )
      contriby.append( y1 )
      contriborient.append( orient1 )
      contribwt.append( mag * (      dx) * (      dy) * (      dorient) )

      # save ram
      del dx, dy, dorient, x0, y0, orient0, x1, y1, orient1
      
      # These are currently lists of arrays.  Concat into one array.
      contribx = np.concatenate( contribx )
      contriby = np.concatenate( contriby )
      contriborient = np.concatenate( contriborient )
      contribwt = np.concatenate( contribwt )

      # Don't allow those lower-right out-of-bounds contributions
      good = np.logical_and( contribx < histdims[1], contriby < histdims[0] )

      indsLinearInt = np.ravel_multi_index( \
        np.vstack([ contriby[good], contribx[good], contriborient[good] ]), \
          histdims )

      # Accumulate all these using bincount.
      orientation_histogram = np.reshape( \
          np.bincount( indsLinearInt,\
                       contribwt[good],\
                       minlength=histnumel), \
            histdims )
      
      # clean up
      del contribx, contriby, contriborient, contribwt, good, indsLinearInt

    # now for each cell, compute the histogram
    hog_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        hog_image = np.zeros((sy, sx), dtype=pedocc.ftype)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / orientations * np.pi)
                    dy = radius * sin(float(o) / orientations * np.pi)
                    rr, cc = draw.line(int(centre[0] - dx),
                                       int(centre[1] - dy),
                                       int(centre[0] + dx),
                                       int(centre[1] + dy))
                    hog_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    eps = (1e-3)**2
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            # JRS: copy is important or it will modify in place
            block = orientation_histogram[y:y + by, x:x + bx, :].copy()
            # JRS: major bug here should be
            #normalised_blocks[y, x, :] = block / sqrt((block**2).sum() + eps)

            # Change to L2-hyst
            block /= sqrt((block**2).sum() + eps)
            # Clip
            vmax = 0.2
            block[ block > vmax ] = vmax
            # normalise again
            block /= sqrt((block**2).sum() + eps)
            # and assign
            normalised_blocks[y, x, :] = block

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if flatten:
      rval = normalised_blocks.ravel()
    else:
      rval = normalised_blocks

    if visualise:
        return rval, hog_image
    else:
        return rval


################################################################################
# Tests

def testGradient( valueType, maxval ):
  print 'Gradient test: type = ', valueType
  tol = 1E-10

  imgR = np.array([
      200, 200, 200, 200, 100, 100, 100, 100,
      200, 200, 200, 200, 100, 100, 100, 100,
      200, 200, 200, 200, 100, 100, 100, 100,
      200, 200, 200, 200, 100, 100, 100, 200,
      100, 100, 100, 100, 100, 100, 200, 200,
      100, 100, 100, 100, 100, 200, 200, 200,
      100, 100, 100, 100, 200, 200, 200, 200,
      100, 100, 100, 200, 200, 200, 200, 200,
      ], dtype=valueType).reshape( (8,8) )

  gx, gy = computeGradients( imgR, False )
  gxGT = np.array([
        0,   0,   0,-100,-100,   0,   0,   0,
        0,   0,   0,-100,-100,   0,   0,   0,
        0,   0,   0,-100,-100,   0,   0,   0,
        0,   0,   0,-100,-100,   0, 100,   0,
        0,   0,   0,   0,   0, 100, 100,   0,
        0,   0,   0,   0, 100, 100,   0,   0,
        0,   0,   0, 100, 100,   0,   0,   0,
        0,   0, 100, 100,   0,   0,   0,   0,
      ], dtype=pedocc.ftype).reshape( (8,8) ) / maxval
  assert np.all( np.abs(gx - gxGT) < tol ), 'diff = %s' % (str( np.abs(gx - gxGT) )) 
  gyGT = np.array([
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0, 100,
     -100,-100,-100,-100,   0,   0, 100, 100,
     -100,-100,-100,-100,   0, 100, 100,   0,
        0,   0,   0,   0, 100, 100,   0,   0,
        0,   0,   0, 100, 100,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
      ], dtype=pedocc.ftype).reshape( (8,8) ) / maxval
  assert np.all( np.abs(gy - gyGT) < tol )
      
  imgG = imgR.copy() / 2
  gx2, gy2 = computeGradients( imgG, False )
  assert np.all( 2*gx2 == gx )
  assert np.all( 2*gy2 == gy )

  # Test that in an RGB image it's the max over channels.  
  gx3, gy3 = computeGradients( np.dstack( [imgR, imgG, imgR] ), False )
  # R channel dominates
  assert np.all( gx3 == gx )
  assert np.all( gy3 == gy )

  # todo: do we want to do [-1,1] gradient at the edges?

if __name__ == "__main__":
  print 'myhog.py: unit tests'
  testGradient( 'uint8', 255 )
  testGradient( 'uint16', 65535 )

  #
  # HOG
  #


