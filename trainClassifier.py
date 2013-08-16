#!/usr/bin/env python
import sys
import pomio

# Usage:
#
#   trainClassifier.py <ftrs.pkl|csv> <labels.pkl|csv> <outfile.pkl> <clfrType> 
#

infileFtrs = sys.argv[1]
infileLabs = sys.argv[2]
outfile    = sys.argv[3]
clfrType   = sys.argv[4]

assert outfile.endswith('.pkl')

assert clfrType in ['logreg', 'randyforest'], 'Unknown classifier type ' + outfileType

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
