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
parser.add_argument('--rf_max_features', type=str, default='auto',  help='max features used in a split.  Can be int, auto, or None')
parser.add_argument('--rf_min_samples_leaf', type=int, default=10,  help='min samples in a tree leaf')
parser.add_argument('--rf_min_samples_split', type=int, default=100,  help='min nb samples to split a node')
parser.add_argument('--ftrsTest', type=str, help='optional test set features for generalisation evaluation')
parser.add_argument('--labsTest', type=str, help='optional test set labels for generalisation evaluation')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nbJobs', type=int, default=-1, \
                        help='Number of parallel jobs during RF training.  -1 to use all available cores.')

args = parser.parse_args()
if args.rf_max_features == 'None':
  args.rf_max_features = None
elif args.rf_max_features != 'auto':
  args.rf_max_features = int( args.rf_max_features )

if args.rf_max_depth == 'None':
  args.rf_max_depth = None
else:
  args.rf_max_depth = int( args.rf_max_depth )

# This is here because something is using gst, which uses arse parser, and that parser is sucking up the -h
import sys
import pomio
import sklearn.ensemble
import sklearn.linear_model
from sklearn import grid_search, cross_validation
import numpy as np
import matplotlib.pyplot as plt

def accuracyPerClass( labsGT, labsPred ):
  n = pomio.getNumClasses()
  correct = (labsGT == labsPred)
  res = np.zeros( (n,), dtype=float )
  for i in range(n):
    msk = labsGT == i
    if np.any(msk):
      res[i] = np.mean( correct[ labsGT == i ] )
  return res

def reportAccuracy( exptName, labs, predlabs ):
  print exptName, ' accuracy (frac correct) = ', np.mean(predlabs==labs)
  apc = accuracyPerClass( labs, predlabs )
  print '   - average accuracy per class = ', apc.mean()
  for i in range(pomio.getNumClasses()):
    print '      %s: %f' %( pomio.getClasses()[i], apc[i] )
  clp, clnum = classProportions( labs )
  print '   - class proportions in %s:' % exptName
  for i in range(pomio.getNumClasses()):
    print '      %15s: %.6f (%6d examples)' %( pomio.getClasses()[i], clp[i], clnum[i] )

def classProportions( labs ):
  res = np.histogram( labs, bins=range(pomio.getNumClasses()+1) )[0].astype(float)
  s = res.sum()
  if s>0:
    prop = res / s
  else:
    prop = np.zeros( (pomio.getNumClasses(),), dtype=float )
  return prop, res

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
D = ftrs.shape[1]
print 'Feature dimensionality = ', D

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
        params['min_samples_leaf'] = [2, 5, 20,100]
        # Not much point searching for n_estimators, bigger is always better,
        # though diminishing returns.
        params['n_estimators']     = [100]
        params['max_depth']        = [15,30,60]
        params['max_depth'].append( None )
        params['max_features']        = [ z for z in [5,15,75] if z<=D ]
        params['max_features'].append( 'auto' )
        params['max_features'].append( None )
        params['min_samples_split'] = [5,10,100]

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
        rfParams['max_features']     = args.rf_max_features if isinstance(args.rf_max_features,str) \
            else min( args.rf_max_features, D )
        rfParams['min_samples_split']= args.rf_min_samples_split

        # some of these might be int
        for k,v in rfParams.items():
          print 'param ', k, ' = ', v, ', type = ', type(rfParams[k])
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
    print 'OOB training set score = ', clfr.oob_score_
elif clfrType == 'logreg':
    print " Give it up for Reggie Log!"
    clfr = sklearn.linear_model.LogisticRegression(penalty='l1' , dual=False, tol=0.0001, C=0.5, fit_intercept=True, intercept_scaling=1)
    clfr = clfr.fit(ftrs, labs)

else:
    print 'Unsupported classifier "', clfrType, '"'
    sys.exit(1)

print '   done.'


def getlabs( clfr, ftrs ):
  if 1:
    predlabs = clfr.predict(ftrs)
  else:
    # This was a quick hack to see if normalising by class prior probability could improve the random forest result.
    # Get probabilities
    probs = clfr.predict_proba( ftrs ) # n x C
    # Normalise by class distn
    priors =np.array([
            0.113521,
            0.189500,
            0.075202,
            0.032654,
            0.022880,
            0.099562,
            0.017276,
            0.086172,
            0.019368,
            0.035853,
            0.026916,
            0.024704,
            0.020982,
            0.013674,
            0.052411,
            0.018023,
            0.092882,
            0.016335,
            0.014287,
            0.020579,
            0.007218,])
    probs /= priors
    # Turn to labs
    predlabs = probs.argmax( axis=1 )

  assert predlabs.ndim==1 and predlabs.shape[0] == ftrs.shape[0]
  return predlabs

predlabs = getlabs(clfr,ftrs)
reportAccuracy( 'Training set', labs, predlabs )

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

    predlabs = getlabs(clfr,ftrsTest)
    reportAccuracy( 'Test set', labsTest, predlabs )

# Write the classifier
if clfr != None and outfile != None:
    pomio.pickleObject( clfr, outfile )
    print 'Output written to file ', outfile
else:
    print "No classifier to persist or no output filename; review input parameters."

