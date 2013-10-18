#!/bin/bash

function Usage() {
    echo "Usage:"
    echo "   ./createMSRCFeatures.sh <MSRCDataPath> <outfileBase>"
}

dataPath="$1"; shift
outfileBase="$1"; shift

if [ ! -d "$dataPath" ]; then
    echo "Error: MSRC path does not exist"
    Usage
    exit 1
fi
if [ ! -d $(dirname "$outfileBase") ]; then
    echo "Error: Outfile directory base $(dirname "$outfileBase") does not exist"
    exit 1
fi


set -e

# Creates feature sets from the MSRC data for a range of parameter values, one for each combination.
# Currently the parameters are for oversegmentation.
spNumVals="400 1000"
spCompactVals="10.0 15.0 30.0"

for spNum in $spNumVals; do
    for spCompact in $spCompactVals; do
	outfileBaseCombo=$(printf "%s_slic-%d-%06.2f" "$outfileBase" $spNum $spCompact)
	echo "*** Combo: spNum = $spNum, spCompact = $spCompact, outfileBase = $outfileBaseCombo"

	echo ./createFeatures.py --type=pkl --nbSuperPixels=$spNum --superPixelCompactness=$spCompact "$dataPath" "$outfileBaseCombo"
    done
done