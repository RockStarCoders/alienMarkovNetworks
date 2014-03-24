import numpy as np
import matplotlib.pyplot as plt
import colorsys

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

    
