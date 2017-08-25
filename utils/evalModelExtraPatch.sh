#!/bin/bash
modelpath="analysis/models/EXP3D_1x1x1x1_0_1_dice_1_best.pth"

for i in {0..50..2};
do
	python eval.py --useGPU=1 --singleEval --modelPath=$modelpath --extraPatch=${i} && \
	echo "Done evaluating "${i}""
done