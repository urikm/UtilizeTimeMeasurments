#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
for i in 1 2
do
	mkdir -p "Results/Analysis_$i"
        taskset -c $((($i-1)*5)),$((($i-1)*5+1)),$((($i-1)*5+2)),$((($i-1)*5+3)),$((($i-1)*5+4)) python3 RunRNeepAnalysis.py --save-path "Results/Analysis_$i" > "Results/Analysis_$i/log.txt" &
done
echo "All the Non Cheatty runs are set and running"
