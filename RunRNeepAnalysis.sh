#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for i in 1
do
	mkdir -p "Results/LR2Analysis_$i"
	python3 RunRNeepAnalysis.py --save-path "Results/LR2Analysis_$i" > "Results/LR2Analysis_$i/log.txt" &
done
echo "All the Non Cheatty runs are set and running"
