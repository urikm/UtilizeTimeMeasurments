#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for i in 1 2 3 4 5
do
	mkdir -p "Results/AnalysisMM_$i"
        python3 RunRNeepAnalysisMM.py --save-path "Results/AnalysisMM_$i" > "Results/AnalysisMM_$i/log.txt"
done
echo "All the runs are set and running"
