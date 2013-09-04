#!/usr/bin/env python
import sys
import pomio
import sklearn.ensemble
import sklearn.linear_model
from sklearn import grid_search, cross_validation
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train a classifier for MRF project.')

parser.add_argument('ftrs', type=str, action='store', \
                        help='filename of pkl or csv training features data')
parser.add_argument('labs', type=str, action='store', \
                        help='filename of pkl or csv training labels data')
parser.add_argument('outfile', type=str, action='store', \
                        help='filename of pkl for output trained classifier object')
parser.add_argument('--type', type=str, action='store', default='randyforest', \
                        choices = ['logreg', 'randyforest'], \
                        help='type of classifier')
parser.add_argument('--paramSearchFolds', type=int, action='store', default=0, \
                        help='number of cross-validation folds for grid search.  0 for no grid search.')

# rf options
parser.add_argument('--rf_n_estimators', type=int, default=50,  help='nb trees in forest')
parser.add_argument('--rf_max_depth',    type=str, default='None',  help='max depth of trees')
parser.add_argument('--rf_min_samples_leaf', type=str, default='10',  help='min samples in a tree leaf')

args = parser.parse_args()

infileFtrs = args.ftrs
infileLabs = args.labs
outfile    = args.outfile
clfrType   = args.type
paramSearchFolds = args.paramSearchFolds

paramSearch = (paramSearchFolds>0)

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

assert np.all( np.isfinite( ftrs ) )

# Train the classifier, either with CV param search or with default values
if paramSearch:

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
        params['min_samples_leaf'] = [5, 10]#list(np.arange(  1,           52, 20, int ))
        params['n_estimators']     = [10,50]#list( np.logspace(1,np.log10(500),  3 ).astype(int) )
        params['max_depth']        = [5]#list( np.arange(  5,           31, 15,int) )
        params['max_depth'].append( None )

        print "\nRandyforest parameter search grid:\n" , params
        
        
        # create classifier and gridsearch classifier
        rf = sklearn.ensemble.RandomForestClassifier()
        gsearch = grid_search.GridSearchCV(rf, params, cv=stratCV, refit=True, verbose=10, n_jobs=-1)

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

    # no grid search, use defaults    
    print '\nUsing default/given params'
    if clfrType == 'randyforest':
        rfParams = {}
        rfParams['min_samples_leaf'] = args.rf_min_samples_leaf
        rfParams['n_estimators']     = args.rf_n_estimators
        rfParams['max_depth']        = args.rf_max_depth
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


print '\nTraining %s classifier on %d examples with deafult param values...' % (clfrType, n)
if clfrType == 'randyforest':
    print '   Introducing Britains hottest rock performer, Randy Forest!'
    clfr = sklearn.ensemble.RandomForestClassifier(\
            max_depth=rfParams['max_depth'],\
            n_estimators=rfParams['n_estimators'], \
            criterion='gini', \
            max_features='auto', \
            min_samples_split=100, \
            min_samples_leaf =rfParams['min_samples_leaf'],\
            bootstrap=True, \
            oob_score=True,\
            n_jobs=-1,\
            random_state=None,\
            verbose=1)

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

