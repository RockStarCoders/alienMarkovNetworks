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
    echo "   ./classifyAllImages.sh <classifierFilename> <outDir> file1 file2 .... fileN"
}

clfrName="$1"; shift
outDir="$1"; shift

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
echo "" > "$csvFn"

logFn="$outDir"/log.txt
echo "" > "$logFn"

for file in $*; do 
    echo "Processing input image $file..."
    echo "*" >> "$logFn"
    echo "* Processing input image $file..." >> "$logFn"
    echo "*" >> "$logFn"
    
    ifn=$(basename "$file")
    extn="${ifn##*.}"
    ifnBase="${ifn%.*}"
    ofn="$outDir"/"$ifn"
    ./testClassifier.py "$clfrName" "$file" --outfile "$ofn" >> "$logFn" 2>&1

    # append to csv
    echo "${ofn},${ifnBase}_GT.$extn" >> "$csvFn"
done

echo "All done"
