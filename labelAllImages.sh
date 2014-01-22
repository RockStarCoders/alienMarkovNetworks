#!/bin/bash

# Usage:
#
#    labelAllImages.sh <classifierFilename> <adjacencyStatsFilename> K 
#                         <outDir> file1 file2 .... fileN
#
# Applies superpixel classifier and MRF to each of the images and puts
# the labelling result in the output dir.  Absolute path for outDir
# recommended.
#
# Example:
#    ./labelAllImages \
#           /vagrant/classifier_msrc_rf_400-10_grid.pkl \
#           /vagrant/features/msrcTraining_slic-400-010.00_adj.pkl \
#           0.1 \
#           /vagrant/results/imagesLabelled  \
#           /vagrant/msrcData/training/Images/*.bmp
#

function Usage() {
    echo "Usage:"
    echo "   ./labelAllImages.sh <classifierFilename> <adjacencyStatsFilename> K <outDir> file1 file2 .... fileN"
}

clfrName="$1"; shift
adjName="$1"; shift
K="$1"; shift
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

echo "MRF Smoothness K = $K"

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

    ./sceneLabelSuperPixels.py "$file" \
	--clfrFn="$clfrName" \
	--adjFn="$adjName" \
	--nbrPotentialMethod=adjacencyAndDegreeSensitive  \
	--nbSuperPixels=400 --superPixelCompactness=10 --K=$K \
	--outfile="$ofn"

    # append to csv
    echo "${ofn},${ifnBase}_GT.$extn" >> "$csvFn"
done

echo "All done"
