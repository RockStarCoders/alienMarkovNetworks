"""
Functionality to do with classifiers, pixel-based and super-pixel-based.
"""
import features
import numpy as np
import pomio
import scipy
import matplotlib.pyplot as plt
import superPixels
import pdb

showDodgySPs = False
if showDodgySPs:
  plt.interactive(1)

def classLabelsOfFeatures( features, classifier ):
  return classifier.predict( features )

# IMPORTANT: there is a column for each CLASS, not each LABEL.  Void is not in there.
# You'll need to offset the indices by 1 to compensate.
def classProbsOfFeatures( features, classifier ):
    assert np.all( classifier.classes_ == np.arange( pomio.getNumClasses() ) ), \
        'Error: given classifier  has %d classes, expecting %d - %s' % \
        ( len(classifier.classes_), pomio.getNumClasses(), str(classifier.classes_) )
    probs = classifier.predict_proba( features )
    # Can't happen now due to above assertion
    # if len(classifier.classes_) != pomio.getNumClasses():
    #     # Transform class probs to the correct sized matrix.
    #     nbClasses = pomio.getNumClasses()
    #     n = probs.shape[0]
    #     cpnew = np.zeros( (n, nbClasses) )
    #     for i in range( probs.shape[1] ):
    #         # stuff this set of probs to new label
    #         cpnew[:,classifier.classes_[i]] = probs[:,i] 
    #     probs = cpnew
    #     del cpnew

    assert probs.shape[1] == pomio.getNumClasses()
    assert probs.shape[0] == features.shape[0]
    return probs

def classifyImagePixels( rgbImage, classifier, ftype, makeProbabilities ):
  outProbs = None
  # todo: ftype stored in classifier?
  ftrs = features.computePixelFeatures( rgbImage, ftype )
  labs = classLabelsOfFeatures( ftrs, classifier )
  labs = np.reshape(labs, (rgbImage.shape[0], rgbImage.shape[1]))

  if makeProbabilities:
    outProbs = classProbsOfFeatures( ftrs, classifier )
    outProbs = np.reshape(outProbs, (rgbImage.shape[0], rgbImage.shape[1], outProbs.shape[1] ))

  return (labs,outProbs)


def classifyImageSuperPixels( rgbImage, classifier, superPixelObj, ftype, aggtype, makeProbabilities ):
  outProbs = None

  # Get superpixels
  imgSuperPixelsMask = superPixelObj.m_labels
  imgSuperPixels = superPixelObj.m_nodes
  numberImgSuperPixels = len(imgSuperPixels)
  print "**Image contains", numberImgSuperPixels, "superpixels"

  # Get superpixel features
  # todo: replace with features.computeSuperPixelFeatures JRS
  spFtrs = features.computeSuperPixelFeatures( rgbImage, superPixelObj, ftype, aggtype )
  spLabels = classLabelsOfFeatures( spFtrs, classifier )

  if makeProbabilities:
    outProbs = classProbsOfFeatures( spFtrs, classifier )

  return (spLabels, outProbs)


# Mod of the function in SuperPixelClassifier.py
def assignClassLabelToSuperPixel(superPixelMask, imageClassLabels):
    """This function provides basic logic for setting the overall class label for a superpixel"""
    # just adopt the most frequently occurring class label as the superpixel label
    superPixelConstituentLabels = imageClassLabels[superPixelMask]
    labelCount = scipy.stats.itemfreq(superPixelConstituentLabels)
    # This is nx2 matrix, each row is (label, count).  Sort in descending count order
    if labelCount.shape[0] > 1:
      idx = np.argsort( labelCount[:,1] )[::-1]
      labelCount = labelCount[ idx, : ]
    # If there are enough pixels of the max class, and the second class is not
    # significant, include it.  Otherwise give it void so it gets discarded
    # later on.
    if float(labelCount[0,1])/len(superPixelConstituentLabels) >= 0.5 and \
          ( labelCount.shape[0]<=1 or labelCount[1,1] <= labelCount[0,1]/2 ):
      return int( labelCount[0,0] )
    else:
      raise Exception( 'WARNING: superpixel has conflicting GT class labels (%.3f %% majority).' % \
                         (100.0 * float(labelCount[0,1])/len(superPixelConstituentLabels)) )


def computeSuperPixelLabels( gtImage, superPixelObj ):
  n = superPixelObj.getNumSuperPixels()
  res = pomio.getVoidIdx() * np.ones( (n,), dtype=int )
  c = 0
  for sp in range( n ):
    try:
      res[sp] = assignClassLabelToSuperPixel( superPixelObj.getLabelImage()==sp, gtImage )
    except:
      # Keep a count of the undecided pixels.  Default label is void
      c += 1

  if c > 0:
    print 'WARNING: %d of %d superpixels have conflicting GT class labels. Setting to void.' % (c,len(res))
    if showDodgySPs:
      # Display for debug purposes.  set to false to turn it off.
      superPixels.displayImage( superPixels.generateImageWithSuperPixelBoundaries( pomio.msrc_convertLabelsToRGB(gtImage),\
                                                                                     superPixelObj.m_labels ), \
                                  imgTitle="Bad image graph", orientation="lower" )
      pdb.set_trace()
      plt.waitforbuttonpress()
                                                                                     
  return res

def computeSuperPixelLabelsMulti( gtImages, superPixelObjs ):
  return np.concatenate([ computeSuperPixelLabels( gtImg, spo ) \
                            for gtImg,spo in zip(gtImages,superPixelObjs) ]).astype(int)
