#!/usr/bin/env python
import sys
import pomio
import sklearn.ensemble
import sklearn.linear_model
import numpy as np

# Usage:
#
#   trainClassifier.py <ftrs.pkl|csv> <labels.pkl|csv> <outfile.pkl> <clfrType> 
#

infileFtrs = sys.argv[1]
infileLabs = sys.argv[2]
outfile    = sys.argv[3]
clfrType   = sys.argv[4]

assert outfile.endswith('.pkl')

assert clfrType in ['logreg', 'randyforest'], \
    'Unknown classifier type ' + outfileType

# Check can write these files.
f=open(outfile,'w')
f.close()

# Load the features and labels
if infileFtrs.endswith('.pkl'):
    ftrs = pomio.unpickleObject( infileFtrs )
else:
    ftrs = pomio.readMatFromCSV( infileFtrs )

if infileLabs.endswith('.pkl'):
    labs = pomio.unpickleObject( infileLabs )
else:
    labs = pomio.readMatFromCSV( infileLabs ).astype(np.int32)

n = len(labs)
assert n == ftrs.shape[0], 'Error: there are %d labels and %d features' \
    % ( n, ftrs.shape[0] )

# Train the classifier
print 'Training %s classifier on %d examples...' % (clfrType, n)
if clfrType == 'randyforest':
    print '   Introducing Britains hottest rock performer, Randy Forest!'
    clfr = sklearn.ensemble.RandomForestClassifier(\
        max_depth=15,\
        n_estimators=100)
    clfr = clfr.fit( ftrs, labs )
elif clfrType == 'logreg':
    print " Give it up for Reggie Log!"
    clfr = sklearn.linear_model.LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1)
    clfr = clfr.fit(ftrs, labs)

else:
    print 'Unsupported classifier "', clfrType, '"'
    sys.exit(1)

print '   done.'

print 'Training set accuracy (frac correct) = ', clfr.score( ftrs, labs )

# Write the classifier
pomio.pickleObject( clfr, outfile )
print 'Output written to file ', outfile
