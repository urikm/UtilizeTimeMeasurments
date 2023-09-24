#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
for i in 1 2 3 4 5
do
	mkdir -p "Results/AnalysisMM_$i"
        taskset -c 0,1,2,3,4 python3 RunRNeepAnalysisMM.py --dump-ds "StoredDatasets3" --save-path "Results/AnalysisMM_$i" > "Results/AnalysisMM_$i/log.txt"
done
echo "All the runs are set and running"
