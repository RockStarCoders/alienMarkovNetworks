#!/bin/bash

# Usage:
#
#    classifyAllImages.sh <classifierFilename> <outDir> file1 file2 .... fileN
#
# Applies superpixel classifier to each of the images and puts the labelling result in the output dir.
# Absolute path for outDir recommended
#
# Example:
#    ./classifyAllImages msrcFull_randForest_grid.pkl `pwd`/results/imagesClassified  \
#           ~/data/sceneLabelling/MSRC_ObjCategImageDatabase_v2/Images/*.bmp
#

function Usage() {
    echo "Usage:"
    echo "   ./classifyAllImages.sh <classifierFilename> <outDir> <nbSuperPix> <superPixCompact> <nbCores> file1 file2 .... fileN"
}

clfrName="$1"; shift
outDir="$1"; shift
nbSuperPixels="$1"; shift
superPixelCompactness="$1"; shift
typeset -i nbCores="$1"; shift

echo "Using $nbCores cores"

if [ ! -d "$outDir" ]; then
    echo "Error: output directory $outDir does not exist."
    Usage
    exit 1
fi

if [ ! -f "$clfrName" ]; then
    echo "Error: classifier $clfrName does not exist"
    Usage
    exit 1
fi

# as we go, create a csv from labelled image to GT image
csvFn="$outDir"/evalpairs.csv
rm -f "$csvFn"

logFn="$outDir"/log.txt
rm -f "$logFn"
echo "" > "$logFn"
typeset -i ctr=0

for file in $*; do 
    echo "Processing input image $file..."
    echo "*" >> "$logFn"
    echo "* Processing input image $file..." >> "$logFn"
    echo "*" >> "$logFn"
    
    ifn=$(basename "$file")
    extn="${ifn##*.}"
    ifnBase="${ifn%.*}"
    ofn="$outDir"/"$ifn"
    ./testClassifier.py "$clfrName" "$file" --outfile "$ofn" \
	--nbSuperPixels $nbSuperPixels \
	--superPixelCompactness $superPixelCompactness \
	--outprobsfile "${ofn%.$extn}.pkl" >> "$logFn" 2>&1 &

    # append to csv
    echo "${ofn},${ifnBase}_GT.$extn" >> "$csvFn"

    ctr=$(($ctr+1))

    if [ $(($ctr % $nbCores)) = 0 ]; then
	wait
    fi
done

wait
echo "All done"
