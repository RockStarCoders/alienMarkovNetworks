import numpy as np

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
