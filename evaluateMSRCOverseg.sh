#!/bin/bash

# Purpose is to determine which oversegmentation parameters are best for the
# MSRC data set.  The optimal combo for the whole system would require all
# possible parameters, ie gridsearch over overseg params and classifier training
# params.  But this would take an age, so they are decoupled: for fixed
# classifier params find the best overseg params, then for best overseg params
# find best classifier params.  This script does the first part of that.
#
# It is assumed you've already run createMSRCFeatures.sh to generate lots of
# feature sets with different oversegmentation parameter combinations.  Now we
# just run the classifier training on each of these in turn, and compute
# validation set accuracy.  The best params are the ones that maximise the
# validation set accuracy.
#
# You should save the output of this script, e.g. ... | tee evalLog.txt
#

function Usage() {
    echo "Usage:"
    echo "   ./evaluateMSRCOverseg.sh /path/to/features/msrc"
    echo "Note feature filename base doesn't have Training,Validation in it."
    echo "Assuming those exist though."
}

featuresBase="$1"; shift

if [ -z "$featuresBase" ]; then
    Usage
    exit 1
fi

echo "Data set,                             Training error,  Validation error"
echo "-----------------------------------------------------------------------"
echo ""

for ftrsTrain in "$featuresBase"Training*_ftrs.pkl; do
    ftrsVal="${ftrsTrain/Training/Validation}"
    results=$( ./trainClassifier.py --type=randyforest \
	--paramSearchFolds=0 --rf_n_estimators=50 --rf_max_depth=100 \
	--rf_max_features=25 --rf_min_samples_leaf=5 \
	--ftrsTest="$ftrsVal" --labsTest="${ftrsVal%_ftrs.pkl}"_labs.pkl \
	"$ftrsTrain" "${ftrsTrain%_ftrs.pkl}"_labs.pkl 2>/dev/null | grep accuracy | cut -d '=' -f 2 )

    # display results
    echo "$(basename $ftrsTrain)   $(echo $results | xargs echo)"  
done