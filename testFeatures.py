#!/usr/bin/env python

import pomio
import amntools
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import FeatureGenerator
import matplotlib

# Load all data
print 'Loading all data...'
data = pomio.msrc_loadImages('/home/jamie/data/MSRC_ObjCategImageDatabase_v2', \
                                 ['Images/7_3_s.bmp'])
# get particular image we like
ex = data[0]


plt.figure()
plt.imshow(ex.m_img)
plt.title('original image')

plt.figure()
clrs = [[z/255.0 for z in c[1]] for c in pomio.msrc_classToRGB]
pomio.showLabels( ex.m_gt )
plt.title('ground truth labels' )

print 'unique class labels: ', np.unique(ex.m_gt)

# generate features
imagePixelFeatures = FeatureGenerator.generatePixelFeaturesForImage(ex.m_img)
# For each feature, how many distinct values?
for i in range(imagePixelFeatures.shape[1]):
    print "Feature %d has %d distinct values" \
        % (i, len(np.unique( imagePixelFeatures[:,i])) )

# Plot a selection of features
sel = np.arange(12,29,2)
# sel = range(80,86)

# colours have to be on range 0-1
plt.figure()
# just plot some of the data for clarity
nbpts = 2000
ptsz = 5
whichPts = np.random.choice( imagePixelFeatures.shape[0], nbpts, replace=False)
# python weirdness, can't index from 2 lists at once...
subset = imagePixelFeatures[whichPts,:]
subset = subset[:,sel]

labelSubset = np.reshape(ex.m_gt, (imagePixelFeatures.shape[0],1))[whichPts].squeeze()
print labelSubset.shape

# todo: fix bug, displayed labels not right, car cluster should be lower right
amntools.gplotmatrix( subset,\
                          labelSubset, \
                          #ex.m_gt.flatten(1)[whichPts], \
                          msize=ptsz, classColours=clrs)
plt.show()
