#!/usr/bin/env python
import sys
import SuperPixelClassifier
import pomio
import numpy as np

# Usage:
#
#   createFeatures.py <MSRC data path> <scaleFrac> <outfileBase> <type=csv|pkl>
#

msrcDataDirectory = sys.argv[1]
scaleFrac         = float(sys.argv[2])
outfileBase       = sys.argv[3]
outfileType       = sys.argv[4]

assert outfileType in ['csv', 'pkl'], 'Unknown outfile type ' + outfileType

outfileFtrs = '%s_ftrs.%s' % ( outfileBase, outfileType )
outfileLabs = '%s_labs.%s' % ( outfileBase, outfileType )
# Check can write these files.
f=open(outfileFtrs,'w')
f.close()
f=open(outfileLabs,'w')
f.close()

assert 0 <= scaleFrac and scaleFrac <= 1

superPixelData     = SuperPixelClassifier.getSuperPixelTrainingData(msrcDataDirectory, scaleFrac)
superPixelFeatures = superPixelData[0]
superPixelLabels   = superPixelData[1].astype(np.int32)

assert np.all( np.isfinite( superPixelFeatures ) )

# Output
if outfileType == 'pkl':
    pomio.pickleObject( superPixelFeatures, outfileFtrs )
    pomio.pickleObject( superPixelLabels,   outfileLabs )
elif outfileType == 'csv':
    pomio.writeMatToCSV( superPixelFeatures, outfileFtrs )
    pomio.writeMatToCSV( superPixelLabels,   outfileLabs )
else:
    assert False, 'unknown output file format ' + outfileType

print 'Output written to file ', outfileFtrs, ' and ', outfileLabs
