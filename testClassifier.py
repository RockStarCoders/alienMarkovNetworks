#!/usr/bin/env python
import sys
import pomio
import sklearn.ensemble
import numpy as np
import matplotlib.pyplot as plt
import SuperPixelClassifier
import skimage

# Usage:
#
#   testClassifier.py <clfr.pkl> <infile1> <infile2> ...
#
#
# Good test image:
#
#   cp ~/data/sceneLabelling/MSRC_ObjCategImageDatabase_v2/Images/3_7_s.bmp
#

clfrFn = sys.argv[1]
clfr = pomio.unpickleObject( clfrFn )

infiles = sys.argv[2:]

plt.interactive(1)
plt.figure()
pomio.showClassColours()

plt.figure()

for fn in infiles:
    print 'Classifying file ', fn
    image = skimage.io.imread(fn)
    #print 'heydy ho image type = ', image.dtype
    [spClassPreds, spGraph] = SuperPixelClassifier.predictSuperPixelLabels(clfr, image)
    spClassPredsImage = spGraph.imageFromSuperPixelData( spClassPreds.reshape( (len(spClassPreds),1) ) )

    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(fn)
    plt.subplot(1,2,2)
    #print spClassPredsImage.shape
    pomio.showLabels(spClassPredsImage)

    plt.waitforbuttonpress()

plt.interactive(0)
plt.show()
