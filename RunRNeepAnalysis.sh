#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for i in 1 2 3 4 5
do
	mkdir -p "Results/AnalysisSt_$i"
        python3 RunRNeepAnalysis.py --save-path "Results/AnalysisSt_$i" > "Results/AnalysisSt_$i/log.txt"
done
echo "All the Non Cheatty runs are set and running"
