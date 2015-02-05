#!/bin/bash
K="$1"; shift
typeset -i nbThreads="$1"; shift
inputDir="$1"; shift
probsBase="$1"; shift
odirBase="$1"; shift
K0="$1"; shift

# e.g.
#
#    ./runOnPaulsResults.sh 5.0  8  \
#        /data/adss/Trials/Aerial/ISPRS_WGIII_benchmark_2014/Labeled/top \
#        /data/adss/users/sherrahj/isprs/paulsResults/2014-12-22/baseline_res \
#        /data/adss/users/sherrahj/isprs/ourResults/mrf/pixel

#export K="10.0"
odir="$odirBase/K_${K}"
mkdir -p "$odir"

typeset -i cnt=0

if [ -z "$K0" ]; then
    kzero=$( echo "$K*0.025" | bc -l )
else
    kzero="$K0"
fi

while read ifn; do
    echo "*** JOB $cnt"
#for file in /data/adss/Trials/Aerial/ISPRS_WGIII_benchmark_2014/Labeled/top/top_mosaic_09cm_area*.tif; do 
    file="$inputDir/$ifn"
    num=${file#*area}
    num=${num%.tif}
    echo $file: $num
    #echo ./sceneLabelN.py \
    ofile="$odir"/$(basename "$file")
    ofile="${ofile%.tif}.png"
    time ./sceneLabelN.py \
	--matFn "$probsBase"_area${num}.mat \
	"$file" --outfile "$ofile" \
	--K $K --nhoodSz 8 --K0 $kzero &
    cnt=$(($cnt+1))
    if [ $cnt == $nbThreads ]; then
	wait
	cnt=0
    fi

done < ~/code/isprs/trainingImages.txt
