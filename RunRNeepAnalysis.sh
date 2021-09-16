#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
for i in 1 2 3 4 5
do
	mkdir -p "Results/Analysis_$i"
	python3 RunRNeepAnalysis.py --save-path "Results/Analysis_$i" > "Results/Analysis_$i/log.txt"
done
echo "All the Non Cheatty runs are set and running"
