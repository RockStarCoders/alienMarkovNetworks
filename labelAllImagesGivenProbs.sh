#!/bin/bash

# Usage:
#
#    labelAllImagesGivenProbs.sh <adjacencyStatsFilename> K 
#                         <outDir> file1 file2 .... fileN
#
# Like labelAllImages.sh, but assumes classifier has already been applied.
# So it works  with the probs not the image, doesn't need classifier or 
# superpixel parameters.
#
# Example:
#    ./labelAllImagesGivenProbs \
#           /vagrant/features/msrcTraining_slic-400-010.00_adj.pkl \
#           0.1 \
#           /vagrant/results/imagesLabelled/training  \
#           /vagrant/results/imagesClassified/training/*.pkl
#

function Usage() {
    echo "Usage:"
    echo "   ./labelAllImages.sh <adjacencyStatsFilename> K <outDir> <nbCores> file1.pkl file2 .... fileN"
}

adjName="$1"; shift
K="$1"; shift
outDir="$1"; shift
typeset -i nbCores="$1"; shift

echo "Using $nbCores cores"

if [ ! -d "$outDir" ]; then
    echo "Error: output directory $outDir does not exist."
    Usage
    exit 1
fi

echo "MRF Smoothness K = $K"

# as we go, create a csv from labelled image to GT image
csvFn="$outDir"/evalpairs.csv
rm -f "$csvFn"

logFn="$outDir"/log.txt
rm -f "$logFn"
typeset -i ctr=0

for file in $*; do 
    echo "Processing input probability pickle file $file..."
    echo "*" >> "$logFn"
    echo "* Processing input probability pickle file $file..." >> "$logFn"
    echo "*" >> "$logFn"
    
    ifn=$(basename "$file")
    extn="${ifn##*.}"
    ifnBase="${ifn%.*}"
    ofn="$outDir"/"$ifnBase".bmp

    ./sceneLabelSuperPixels.py "$file" \
	--adjFn="$adjName" \
	--nbrPotentialMethod=degreeSensitive  \
	--K=$K \
	--outfile="$ofn" >> "$logFn" 2>&1 &

    # append to csv
    echo "${ofn},${ifnBase}_GT.bmp" >> "$csvFn"

    ctr=$(($ctr+1))

    if [ $(($ctr % $nbCores)) = 0 ]; then
	wait
    fi
done

wait
echo "All done"
