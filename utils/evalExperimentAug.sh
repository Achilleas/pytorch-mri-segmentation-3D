#!/bin/bash
modelpath="analysis/models/EXP3D_1x1x1x1_0_1_dice_1_best.pth"
python eval.py --useGPU=1 --singleEval --modelPath=$modelpath --extraPatch=16 --testAugm