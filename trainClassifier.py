#!/usr/bin/env python

"""
Command-line tool for training classifiers on pixel or super-pixel features.
"""

import argparse
parser = argparse.ArgumentParser(description='Train a classifier for MRF project.')

parser.add_argument('ftrs', type=str, action='store', \
                        help='filename of pkl or csv training features data')
parser.add_argument('labs', type=str, action='store', \
                        help='filename of pkl or csv training labels data')
parser.add_argument('--outfile', type=str, action='store', \
                        help='filename of pkl for output trained classifier object')
parser.add_argument('--type', type=str, action='store', default='randyforest', \
                        choices = ['logreg', 'randyforest'], \
                        help='type of classifier')
parser.add_argument('--paramSearchFolds', type=int, action='store', default=0, \
                        help='number of cross-validation folds for grid search.  0 for no grid search.')

# rf options
parser.add_argument('--rf_n_estimators', type=int, default=50,  help='nb trees in forest')
parser.add_argument('--rf_max_depth',    type=str, default='None',  help='max depth of trees')
parser.add_argument('--rf_max_features', type=str, default='auto',  help='max features used in a split')
parser.add_argument('--rf_min_samples_leaf', type=str, default='10',  help='min samples in a tree leaf')
parser.add_argument('--rf_min_samples_split', type=int, default=100,  help='min nb samples to split a node')
parser.add_argument('--ftrsTest', type=str, help='optional test set features for generalisation evaluation')
parser.add_argument('--labsTest', type=str, help='optional test set labels for generalisation evaluation')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nbJobs', type=int, default=-1, \
                        help='Number of parallel jobs during RF training.  -1 to use all available cores.')

args = parser.parse_args()

# This is here because something is using gst, which uses arse parser, and that parser is sucking up the -h
import sys
import pomio
import sklearn.ensemble
import sklearn.linear_model
from sklearn import grid_search, cross_validation
import numpy as np
import matplotlib.pyplot as plt


infileFtrs = args.ftrs
infileLabs = args.labs
outfile    = args.outfile
clfrType   = args.type
paramSearchFolds = args.paramSearchFolds
infileFtrsTest = args.ftrsTest
infileLabsTest = args.labsTest

paramSearch = (paramSearchFolds>0)

assert outfile == None or outfile.endswith('.pkl')

assert clfrType in ['logreg', 'randyforest'], \
    'Unknown classifier type ' + clfrType

# Check can write these files.
if outfile != None:
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
assert n == ftrs.shape[0], 'Error: there are %d labels and %d training examples' \
    % ( n, ftrs.shape[0] )

assert np.all( np.isfinite( ftrs ) )

print 'There are %d unique labels in range [%d,%d]' % ( len(np.unique(labs)), np.min(labs), np.max(labs) )

if args.verbose:
    print 'There are %d training examples' % len(labs)
    plt.interactive(True)
    plt.hist( labs, bins=range(pomio.getNumLabels()) )
    plt.waitforbuttonpress()
    
# Train the classifier, either with CV param search or with default values
if paramSearch:
    paramSrc = 'grid search'
    # create crossValidation object
    stratCV = cross_validation.StratifiedKFold(labs, paramSearchFolds)

    print 'Training %s classifier using %d-fold cross-validation parameter search, over %s samples...' % (clfrType, paramSearchFolds, n)

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
        params['min_samples_leaf'] = [5, 20,100]
        params['n_estimators']     = [50,150,500]
        params['max_depth']        = [15,50]
        params['max_depth'].append( None )
        params['max_features']        = [5,75]
        params['max_features'].append( 'auto' )
        params['min_samples_split'] = [10,100]

        print "\nRandyforest parameter search grid:\n" , params
        
        
        # create classifier and gridsearch classifier
        rf = sklearn.ensemble.RandomForestClassifier()
        gsearch = grid_search.GridSearchCV(rf, params, cv=stratCV, refit=True, verbose=10, n_jobs=args.nbJobs)

        # train grid search on data
        gsearch.fit(ftrs, labs)

        # get best parameters
        rfParams = gsearch.best_params_
        print 'Done.  Grid search gave these parameters:'
        for k,v in rfParams.items():
            print k, ': ', v
    else:
        print 'Unsupported classifier "', clfrType, '"'
        sys.exit(1)

else:
    paramSrc = 'default/specified'
    # no grid search, use defaults    
    print '\nUsing default/given params'
    if clfrType == 'randyforest':
        rfParams = {}
        rfParams['min_samples_leaf'] = args.rf_min_samples_leaf
        rfParams['n_estimators']     = args.rf_n_estimators
        rfParams['max_depth']        = args.rf_max_depth
        rfParams['max_features']     = args.rf_max_features
        rfParams['min_samples_split']= args.rf_min_samples_split

        # some of these might be int
        for k,v in rfParams.items():
            if type(v)==str:
                if v=='None':
                    rfParams[k] = None
                elif v.isdigit():
                    rfParams[k] = int(v)
            print 'param = ', v, ', type = ', type(rfParams[k])
    elif clfrType == 'logreg':
        print " Give it up for Reggie Log!"
        clfr = sklearn.linear_model.LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1)
        clfr = clfr.fit(ftrs, labs)
    else:
        print 'Unsupported classifier "', clfrType, '"'
        sys.exit(1)


print '\nTraining %s classifier on %d examples with %s param values...' % (clfrType, n, paramSrc)
if clfrType == 'randyforest':
    print '   Introducing Britains hottest rock performer, Randy Forest!'
    clfr = sklearn.ensemble.RandomForestClassifier(\
            max_depth=rfParams['max_depth'],\
            n_estimators=rfParams['n_estimators'], \
            criterion='gini', \
            max_features=rfParams['max_features'], \
            min_samples_split=rfParams['min_samples_split'], \
            min_samples_leaf =rfParams['min_samples_leaf'],\
            bootstrap=True, \
            oob_score=True,\
            n_jobs=args.nbJobs,\
            random_state=None,\
            verbose=0)

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

# optionally test classifier on hold-out test set
if infileFtrsTest != None and infileLabsTest != None:
    # Load the features and labels
    if infileFtrsTest.endswith('.pkl'):
        ftrsTest = pomio.unpickleObject( infileFtrsTest )
    else:
        ftrsTest = pomio.readMatFromCSV( infileFtrsTest )
    
    if infileLabsTest.endswith('.pkl'):
        labsTest = pomio.unpickleObject( infileLabsTest )
    else:
        labsTest = pomio.readMatFromCSV( infileLabsTest ).astype(np.int32)
    
    ntest = len(labsTest)
    assert ntest == ftrsTest.shape[0], 'Error: for TEST set, there are %d labels and %d features' \
        % ( ntest, ftrsTest.shape[0] )
    
    assert np.all( np.isfinite( ftrsTest ) )

    print 'Test set accuracy (frac correct)     = ', clfr.score( ftrsTest, labsTest )

# Write the classifier
if clfr != None and outfile != None:
    pomio.pickleObject( clfr, outfile )
    print 'Output written to file ', outfile
else:
    print "No classifier to persist or no output filename; review input parameters."

