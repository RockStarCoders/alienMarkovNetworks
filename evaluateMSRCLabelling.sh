#!/bin/bash

#Kvals="0.01 0.05 0.1 0.5 1.0 1.5 2.0"
Kvals="0.1 0.2 0.3 0.4 0.5"

# assume you have called classifyAllImages on validation set to make image
# probabilities as pkl files.
#
# Give this directory as an input to this script.  Will process all pkl files as
# input.

function Usage() {
    echo "Usage:"
    echo "   ./evaluateMSRCLabelling.sh <indir> <outdir> <MSRCPath> <adjFile>"
    echo "indir is where the input probability pkl files are."
    echo "MSRCPath is where the ground truth data is."
    echo "WARNING: USE ABSOLUTE PATHS!"
}

inDir="$1"; shift
outDirBase="$1"; shift
msrcPath="$1"; shift
adjFile="$1"; shift

if [ ! -d "$inDir" ]; then
    echo "Error: input directory $inDir DNE."
    Usage
    exit 1
fi
if [ ! -d "$outDirBase" ]; then
    echo "Error: output directory $outDirBase DNE."
    Usage
    exit 1
fi
if [ ! -d "$msrcPath" ]; then
    echo "Error: msrc path $msrcPath DNE."
    Usage
    exit 1
fi
if [ ! -f "$adjFile" ]; then
    echo "Error: class adjacency file $adjFile DNE."
    Usage
    exit 1
fi 

set -e

for K in $Kvals; do
    echo "Evaluting MRF for K = $K"

    outDir="$outDirBase/Kis${K}"
    mkdir "$outDir"

    #for each MSRC class probabilities image in validation set:
    ./labelAllImagesGivenProbs.sh "$adjFile" "$K" "$outDir" "$inDir"/*.pkl \
	>/dev/null 2>/dev/null

    # evaluate output labellings against GT
    #echo ./evalPredictions.py "$outDir"/evalpairs.csv "$msrcPath" ''
    acc=$( ( ./evalPredictions.py "$outDir"/evalpairs.csv "$msrcPath" '' 2>/dev/null ) | grep 'Avg prediction' | cut -d '=' -f 2 )

    echo $acc
    # report accuracy for this K
    echo "  K = $K, average accuracy = $acc"
done
