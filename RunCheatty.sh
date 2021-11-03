#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for i in 1 3 4
do
	mkdir -p "Results/NonReduced_Analysis_$i"
	python3 RunCheatty.py --save-path "Results/NonReduced_Analysis_$i" > "Results/NonReduced_Analysis_$i/log.txt"
done
echo "All the Cheatty runs are set and running"
