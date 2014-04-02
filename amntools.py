import numpy as np
import matplotlib.pyplot as plt
import colorsys
import pickle
import pomio

"""
Miscellaneous Tools
"""

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def estimateNeighbourRMSPixelDiff(imgRGB, nhoodSz):
    # estimate neighbourhood average diff
    sigsq = 0
    cnt = 0
    rows = imgRGB.shape[0]
    cols = imgRGB.shape[1]
    # vertical diffs
    idiffs = imgRGB[0:rows-1,:,:] - imgRGB[1:rows,:,:]
    sigsq += np.power(idiffs,2.0).sum()
    cnt += (rows-1)*cols
    # horizontal diffs
    idiffs = imgRGB[:,0:cols-1,:] - imgRGB[:,1:cols,:]
    sigsq += np.power(idiffs,2.0).sum()
    cnt += rows*(cols-1)
    
    if nhoodSz == 8:
        # diagonal to right
        idiffs = imgRGB[0:rows-1,0:cols-1,:] - imgRGB[1:rows,1:cols,:]
        sigsq += np.power(idiffs,2.0).sum()
        cnt += (rows-1)*(cols-1)
        # diagonal to left
        idiffs = imgRGB[0:rows-1,1:cols,:] - imgRGB[1:rows,0:cols-1,:]
        sigsq += np.power(idiffs,2.0).sum()
        cnt += (rows-1)*(cols-1)
    
    sigsq /= cnt
    return np.sqrt(sigsq)


def gplotmatrix( X, labels, msize=5, classColours=None, featureNames=None ):
    assert X.ndim == 2
    D = X.shape[1]
    print labels
    assert type(labels)==list or labels.ndim == 1
    assert( len(labels) == X.shape[0] )
    # assume labels are contiguous
    lmin = np.min(labels)
    lmax = np.max(labels)
    plt.clf()
    if classColours == None:
        classColours = _get_colors(lmax+1)

    idx = 1
    for r in range(D):
        for c in range(D):
            plt.subplot(D,D,idx)
            idx += 1
            x1 = X[:,r]
            x2 = X[:,c]
            if r==c:
                # histogram
                plt.hist( x1 )
            else:
                for l in np.unique(labels):#range(lmin,lmax+1):
                    #print labels==l
                    plt.plot( x2[labels==l], x1[labels==l], '.', \
                                  color=classColours[l],\
                                  markersize=msize)
                    #print 'foo'
                    #plt.waitforbuttonpress()
                    plt.hold(1)
                plt.hold(0)
            plt.grid(1)
            if c==0:
              sl = str(r)
              if featureNames != None:
                sl += ': ' + featureNames[r]
              plt.ylabel( sl )
            if r==D-1:
              sl = str(c)
              if featureNames != None:
                sl += ': ' + featureNames[c]
              plt.xlabel( sl )

    

# Features is nxd matrix
def classifyFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has %d classes - %s' % \
            ( len(classifier.classes_), str(classifier.classes_) )
    c = classifier.predict( features )
    return c

# IMPORTANT: there is a column for each CLASS, not each LABEL.  Void is not in there.
# You'll need to offset the indices by 1 to compensate.
def classProbsOfFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has %d classes - %s' % \
            ( len(classifier.classes_), str(classifier.classes_) )
    probs = classifier.predict_proba( features )
    if len(classifier.classes_) != pomio.getNumClasses():
        # Transform class probs to the correct sized matrix.
        nbClasses = pomio.getNumClasses()
        n = probs.shape[0]
        cpnew = np.zeros( (n, nbClasses) )
        for i in range( probs.shape[1] ):
            # stuff this set of probs to new label
            cpnew[:,classifier.classes_[i]-1] = probs[:,i] 
        probs = cpnew
        del cpnew

    assert probs.shape[1] == pomio.getNumClasses()
    return probs
