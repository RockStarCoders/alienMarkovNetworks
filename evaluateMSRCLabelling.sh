#!/bin/bash

Kvals="0.01 0.05 0.1 0.5 1.0 10.0"

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
outDir="$1"; shift
msrcPath="$1"; shift
adjFile="$1"; shift

if [ ! -d "$inDir" ]; then
    echo "Error: input directory $inDir DNE."
    Usage
    exit 1
fi
if [ ! -d "$outDir" ]; then
    echo "Error: output directory $outDir DNE."
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

# as we go, create a csv from labelled image to GT image
csvFn="$outDir"/evalpairs.csv
rm -f "$csvFn"

logFn="$outDir"/log.txt
rm -f "$logFn"


for K in $Kvals; do
    echo "Evaluting MRF for K = $K"
    #for each MSRC class probabilities image in validation set:
    for probsfile in "$inDir"/*.pkl; do
       #smooth labels with MRF given K
       #output labelling
	ifn=$(basename "$probsfile")
	ifnBase="${ifn%.*}"
	ofn="$outDir/${ifnBase}.png"
	#echo "  * Processing file $probsfile --> $ofn"

        # append to csv
	echo "${ofn},${ifnBase}_GT.bmp" >> "$csvFn"

	./sceneLabelSuperPixels.py \
	    --adjFn "$adjFile" \
	    --K $K \
	    --nbrPotentialMethod adjacencyAndDegreeSensitive \
	    "$probsfile" \
	    --outfile "$ofn" >> "$logFn" 2>&1


    done
    # evaluate output labellings against GT
    acc=$(./evalPredictions.py "$outDir"/evalpairs.csv "$msrcPath" "" | grep "**Avg prediction accuracy" | \
	cut -d '=' -f 2)

    # report accuracy for this K
    echo "  K = $K, average accuracy = $acc"
done
