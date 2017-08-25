#!/bin/bash

experimentModelsPath="analysis/models/*"

for i in $(ls -d $experimentModelsPath);
do
	python eval.py --useGPU=1 --singleEval --modelPath=${i} && \
	echo "Done evaluating "${i}""
done