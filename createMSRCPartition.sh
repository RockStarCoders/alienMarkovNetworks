#!/bin/bash

# This script creates the partition of MSRC data into training, validation and
# test images published by Shotton and used by others.  To be consistent with
# the literature so comparisons are valid.
#
# For generality, it's all done through subdirectories and sym links.  You
# provide the source data directory for the data set, make sure you use an
# absolute path.  Then you provide a destination.  The default is:
#
#     <repoPath>/vagrant/msrcData
#
# so that when you use a VM, the data is accessible from /vagrant on the VM.
#
# For the code to work it assumes the MSRC directory structure, this is
# recreated for each of the training, validation and test set:
#
#    <destPath>/training/Images/
#    <destPath>/training/GroundTruth/
#    <destPath>/training/SegmentationsGTHighQuality/
#
#    <destPath>/validation/Images/
#    <destPath>/validation/GroundTruth/
#    <destPath>/validation/SegmentationsGTHighQuality/
#
#    <destPath>/test/Images/
#    <destPath>/test/GroundTruth/
#    <destPath>/test/SegmentationsGTHighQuality/
#
# On the command line you also provide the three text files containing the input
# images for each of training, validation and test.  The format is path-less
# filenames of the images, one per line:
#
#    16_11_s.bmp
#    15_24_s.bmp
#    8_25_s.bmp
#    2_15_s.bmp
#    11_22_s.bmp
#    ...
#
#
# Note: ignoring high quality gt for now.

function Usage() {
    echo "Usage:"
    echo "   ./createMSRCPartition.sh [-c] <MSRCDataPath> <trList.txt> <valList.txt> <tstList.txt> <destPath>"
    echo "If the optional -c flag is given, then the files are copied instead of symlinked."
}

if [ "$1" == "-c" ]; then
    doCopy="true"
    echo "COPY MODE"
    shift
fi

typeset -a dataSets=('training' 'validation' 'test')

msrcPath="$1"; shift
if [ ! -d "$msrcPath" ]; then
    echo "Error: MSRC path does not exist"
    Usage
    exit 1
fi

typeset -a fileLists=()
for (( i=0; i<3; ++i )) {
    infile="$1"; shift
    fileLists[$i]="$infile"
    if [ ! -f "$infile" ]; then
	echo "Error: ${dataSets[$i]} set list file $infile does not exist."
	Usage
	exit 1
    fi
}

destPath="$1"; shift
if [ -d "$destPath" ]; then
    echo "Error: Destination path $destPath areadly exists.  Exiting"
    exit 1
fi
if [ ! -d $(dirname "$destPath") ]; then
    echo "Error: Destination directory base $(dirname "$destPath") does not exist"
    exit 1
fi

# no errors from now on
set -e

echo "Source data set at $msrcPath"
echo "Making outdir $destPath"
mkdir "$destPath"

for (( i=0; i<3; ++i )) {
    dsname=${dataSets[$i]}
    odir="$destPath/$dsname"
    echo "*** Creating $dsname set under $odir"
    mkdir -p "$odir/Images"
    mkdir -p "$odir/GroundTruth"
    #mkdir -p "$odir/SegmentationsGTHighQuality"
    # Iterate over files in our file list
    listFile="${fileLists[$i]}"
    while read imgFile; do 
	bfn=$(basename $imgFile)
	if [ -z "$doCopy" ]; then
	    ln -s "$msrcPath/Images/$imgFile" "$odir/Images/"
	    ln -s "$msrcPath/GroundTruth/${imgFile%.bmp}_GT.bmp" "$odir/GroundTruth/"
	else
	    cp "$msrcPath/Images/$imgFile" "$odir/Images/"
	    cp "$msrcPath/GroundTruth/${imgFile%.bmp}_GT.bmp" "$odir/GroundTruth/"
	fi
    done < "$listFile"
    echo ""
}

# Finally construct the trainingPlusValidation directory.
echo "*** Creating trainingPlusValidation directory..."
cp -r "$destPath"/training "$destPath"/trainingPlusValidation
cp -r "$destPath"/validation "$destPath"/trainingPlusValidation

echo "Finito"
