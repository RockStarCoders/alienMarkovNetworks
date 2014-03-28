#!/bin/bash

# The purpose is to reproduce SOA results on the MSRC data set.
#
# You need to do something like this:
#
#     export PYTHONPATH="/home/sherrahj/code/lib/slic-python:/home/sherrahj/code/alienMarkovNetworks/maxflow"

dataPath="$1"; shift
outDir="$1"; shift

if [ -z "$dataPath" -o -z "$outDir" ]; then
    echo "Usage Error:"
    echo "   reproduceMSRC.sh <MSRC data path> <output directory>"
    echo "The output dir must not exist.  The input directory contains"
    echo "the pre-generated train/val/test data sets, see createMSRCPartition.sh"
    echo "and README.md."
    exit 1
fi

set -e

# Create output directory.  It must not already exist.
mkdir "$outDir"

# Steps are:
#   - extract features from images
#   - train classifier
#   - apply classifier to images
#   - apply MRF to classifications
#   - compute accuracy of classifier output, MRF output against GT

# 
# Parameters
# 
# In github see these issues:
#
#   https://github.com/RockStarCoders/alienMarkovNetworks/issues/27
#   https://github.com/RockStarCoders/alienMarkovNetworks/issues/10
#
slicN=400
slicC=10.0
K=0.2
ftrsBase="$outDir"/ftrsTrain

echo "*** Creating training features..."
./createFeatures.py --type=pkl \
    --nbSuperPixels=$slicN --superPixelCompactness=$slicC \
    "$dataPath/trainingPlusValidation" "$ftrsBase"
echo "  done"

echo "*** Training classifier..."
clfrFn="$outDir"/classifier.pkl
./trainClassifier.py --type=randyforest \
    --paramSearchFolds=0 --rf_n_estimators=50 --rf_max_depth=100 \
    --rf_max_features=25 --rf_min_samples_leaf=5 \
    "$ftrsBase_ftrs.pkl" "$ftrsBase%_labs.pkl" \
    --outfile="$clfrFn"
echo "  done"

echo "*** Classify test images..."
oclDir="$outDir"/classified
mkdir "$oclDir"
./classifyAllImages.sh "$clfrFn" "$oclDir" $slicN $slicC \
    "$dataPath"/test/Images/*.bmp     
echo "  done"

echo "*** Label test images..."
olabDir="$outDir"/labelled
mkdir "$olabDir"
./labelAllImagesGivenProbs.sh "$ftrsBase_adj.pkl" $K "$olabDir" "$oclDir"/*.pkl
echo "  done"


echo "*** Evaluating results..."
echo "  * Classifier only: "
./evalPredictions.py "$oclDir"/evalpairs.csv "$dataPath/test" ''
echo "  * Classifier+MRF:  "
./evalPredictions.py "$olabDir"/evalpairs.csv "$dataPath/test" ''
echo "      done"
