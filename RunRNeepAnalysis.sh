#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for i in 1 2 3 4
do
	mkdir -p "Results/AnalysisZoomed_$i"
        CUDA_VISIBLE_DEVICES=$i python3 RunRNeepAnalysis.py --save-path "Results/AnalysisZoomed_$i" > "Results/AnalysisZoomed_$i/log.txt"
done
echo "All the Non Cheatty runs are set and running"
