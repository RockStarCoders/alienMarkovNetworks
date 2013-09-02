#!/usr/bin/env python
import sys
import pomio
import sklearn.ensemble
import sklearn.linear_model
from sklearn import grid_search, cross_validation
import numpy as np

# Usage:
#
#   trainClassifier.py <ftrs.pkl|csv> <labels.pkl|csv> <outfile.pkl> <clfrType> <paramSearch='true','True'|'1'>
#

infileFtrs = sys.argv[1]
infileLabs = sys.argv[2]
outfile    = sys.argv[3]
clfrType   = sys.argv[4]
paramSearch = sys.argv[5]

assert outfile.endswith('.pkl')

assert clfrType in ['logreg', 'randyforest'], \
    'Unknown classifier type ' + outfileType

# Check can write these files.
f=open(outfile,'w')
f.close()


clfr = None
labs = None
ftrs = None


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


# Train the classifier, either with CV param search or with default values
if paramSearch == 'true' or paramSearch == 'True' or paramSearch == '1':

    # create crossValidation object
    stratCV = cross_validation.StratifiedKFold(labs, 10)

    print 'Training %s classifier using 10-fold cross-validation parameter search, over %s samples...' % (clfrType, n)

    # empy param values
    params = {}
    
    if clfrType == 'logreg':
        # create a set of C value and regularisation types for logisitc regression            
        Cmin = 0.0
        Cmax = 100
        inc = 5
        CValues = []

        for C in range(int(Cmin), int(Cmax+inc), inc):
            if C == 0.0 or C == 0:
                C = 0.001
            CValues.append(float(C))

        params['C'] = CValues
    
        # regularisation penalty
        penaltyValues = [ 'l1' , 'l2' ]
        params['penalty'] = penaltyValues

        print "\n\tLogistic regression parameter grid::\n\t" , params
        
        lr = sklearn.linear_model.LogisticRegression()
        clfr = grid_search.GridSearchCV(lr, params, cv=stratCV)
        
        print "\n\tNow fitting data to cross-validation training data across parameter set..."
        # train grid search on data
        clfr.fit(ftrs, labs)

        # get best parameters
        result = clfr.best_estimator_

            
    elif clfrType == 'randyforest':
        # create a set of parameters 
        inc5 = 5
        inc10 = 10
        
        minLeafSamples_min = 10
        minLeafSamples_max = 30
        minLeafSamplesValues = []

        for leafValue in range(minLeafSamples_min , minLeafSamples_max + inc5 , inc5):
            minLeafSamplesValues.append(leafValue)
        
        params['min_samples_leaf'] = minLeafSamplesValues
        
        nEstimators_min = 10
        nEstimators_max = 30
        nEstimatorsValues = []
    
        for nEstValue in range(nEstimators_min, nEstimators_max + inc5, inc5):
            nEstimatorsValues.append(nEstValue)
    
        params['n_estimators'] = nEstimatorsValues
        
        maxDepth_min = 10
        maxDepth_max = 30
        maxDepthValues = []
        
        for maxDepthValue in range(maxDepth_min, maxDepth_max + inc5 , inc5):
            maxDepthValues.append(maxDepthValue)
        
        params['max_depth'] = maxDepthValues
        
        print "\nRandyforest parameter search grid:\n" , params
        
        
        # create classifier and gridsearch classifier
        rf = sklearn.ensemble.RandomForestClassifier()
        clfr = grid_search.GridSearchCV(rf, params, cv=stratCV)

        # train grid search on data
        clfr.fit(ftrs, labs)

        # get best parameters
        result = clfr.best_estimator_
        
    else:
        print 'Unsupported classifier "', clfrType, '"'
        sys.exit(1)

    

else:

    print '\nNo parameter search requested.  \nTraining %s classifier on %d examples with deafult param values...' % (clfrType, n)
    if clfrType == 'randyforest':
        print '   Introducing Britains hottest rock performer, Randy Forest!'
        clfr = sklearn.ensemble.RandomForestClassifier(\
            max_depth=None,\
            n_estimators=20, criterion='gini', max_features='auto', \
                min_samples_split=100, \
                min_samples_leaf =20,\
                bootstrap=True, \
                oob_score=True, n_jobs=-1, random_state=None, verbose=1)
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
if clfr != None:
    pomio.pickleObject( clfr, outfile )
    print 'Output written to file ', outfile
else:
    print "No classifier to persist; review input parameters."

