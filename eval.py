import sys

sys.path.append('architectures/deeplab_3D/')
sys.path.append('architectures/unet_3D/')
sys.path.append('architectures/hrnet_3D/')
sys.path.append('architectures/experiment_nets_3D/')
sys.path.append('utils/')

import os
from os import walk
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.misc
import os
from tqdm import *
import random
from random import randint
from docopt import docopt

import deeplab_resnet_3D
import unet_3D
import highresnet_3D
import exp_net_3D

import lossF
import PP
import augmentations as AUG
import evalF as EF
import evalFP as EFP
import evalMetrics as METRICS

docstr = """Write something here

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --evalMethod=<int>          0 for evaluation of model by whole image, 1 for patches [default: 1]
    --patchPredSize=<int>       If evaluating model with patches, the size of the patch [default: 60]
    --evalMetric=<str>          'iou','dice',only iou supported right now [default: iou]
    --snapPrefix=<str>          Snapshot prefix. a_1000.pth, a_2000.pth, a is prefix [default: HR3Dadice_1_2017-07-16-18-32_iter]
    --singleEval                Evaluate a single model
    --postFix=<str>             Postfix [default: _200x200x100orig]
    --resultsDir=<str>          Path to save evaluation results and predictions to [default: eval_results/]
    --predictionsPath=<str>     predictions path [default: 1]
    --snapshotPath=<str>        Snapshot path [default: models/snapshots/]
    --mainFolderPath=<str>      Main folder path [default: ../Data/MS2017b/]
    --NoLabels=<int>            The number of different labels in training data [default: 2]
    --gpu0=<int>                GPU number [default: 0]
    --useGPU=<int>              Use GPU [default: 0]
    --testMode                  Enable test model (no evaluation, only predictions)
    --modelPath=<str>           Full model path to test if only 1 model (test mode or singleEval mode use this) [default: None]
    --iterRange=<str>           Range of num iters [default: 1-21]
    --iterStep=<int>            Step size of iters [default: 1]
    --testAugm                  Apply test time augmentations
    --extraPatch=<int>          Extra patch size [default: 5]
"""

args = docopt(docstr, version='v0.1')
print(args)

eval_method = int(args['--evalMethod'])
patch_pred_size = int(args['--patchPredSize'])
eval_metric = args['--evalMetric']
snap_prefix = args['--snapPrefix']
results_dir = args['--resultsDir']
snapshots_path = args['--snapshotPath']
postfix = args['--postFix']
main_folder_path = args['--mainFolderPath']
num_labels = int(args['--NoLabels'])
gpu0 = int(args['--gpu0'])
useGPU = int(args['--useGPU'])
test_mode = args['--testMode']
model_path = args['--modelPath']
iter_range = args['--iterRange']
iter_step = int(args['--iterStep'])
iter_low, iter_high = int(iter_range.split('-')[0]), int(iter_range.split('-')[1])
eval_list = main_folder_path + 'val' + postfix + '.txt'
test_augm = args['--testAugm']
single_eval = args['--singleEval']
extra_patch = int(args['--extraPatch'])
if single_eval or test_mode:
    models_path = model_path
else:
    models_path = snap_prefix

if test_mode:
    if not os.path.exists('temp_preds/'):
        os.makedirs('temp_preds/')
else:
    if not os.path.exists(results_dir):
        print('Creating directory at:' , results_dir)
        os.makedirs(results_dir)
    results_file = open(os.path.join(results_dir, os.path.splitext(os.path.basename(models_path))[0] + '.txt'), 'w+')

if num_labels == 2:
    onlyLesions = True
else:
    onlyLesions = False

num_labels2 = 209

def modelInit():
    isPriv = False
    f_name = models_path.split('/')[-1]
    #load model
    if 'EXP3D' in f_name:
        experiment = f_name.replace('EXP3D_', '').replace('.pth', '').split('_')
        experiment = '_'.join(experiment[0:3])
        dilation_arr, isPriv, withASPP = PP.getExperimentInfo(experiment)
        model = exp_net_3D.getExpNet(num_labels, dilation_arr, isPriv, NoLabels2 = num_labels2, withASPP = withASPP)
    elif 'HR3D' in f_name:
        model = highresnet_3D.getHRNet(num_labels)
    elif 'DL3D' in f_name:
        model = deeplab_resnet_3D.Res_Deeplab(num_labels)
    elif 'UNET3D' in  f_name:
        model = unet_3D.UNet3D(1, num_labels)
    else:
        print('No model available for this .pth')
        sys.exit()

    model.eval()

    if useGPU:
        model.cuda(gpu0)

    return model, isPriv

def evalModel(model):
    img_list = open(eval_list).readlines()
    if test_mode:
        if models_path == 'None':
            print('Insert model path if you are testing this model')
            sys.exit()
        model = loadSnapshot(model, models_path)

        for img_str in img_list:
            img_str = img_str.rstrip()
            img, gt, out, affine = EF.predict(os.path.join(main_folder_path, img_str),
                                                        model, num_labels, postfix, main_folder_path, eval_method, 
                                                        gpu0, useGPU, patch_size = patch_pred_size, test_augm = test_augm, extra_patch = extra_patch)

            #save prediction
            save_path = os.path.join('temp_preds', 'pred_' + img_str.split('/')[-3] + '_s' + str(gt.shape[0]) + '.nii.gz')
            PP.saveScan(out, affine, save_path)
    else:
        if single_eval:
            r = range(1)
        else:
            r = range(iter_low, iter_high, iter_step)
        for iter in r:
            counter = 0
            if single_eval:
                model = loadSnapshot(model, models_path)
            else:
                model = loadSnapshot(model, os.path.join(snapshots_path, models_path + '_' + str(iter*1000) + '.pth'))
            r_list_iou = []
            r_list_dice = []
            r_list_recall = []
            r_list_precision = []
            for img_str in img_list:
                img_str = img_str.rstrip()
                img, gt, out, _ = EF.predict(os.path.join(main_folder_path, img_str),
                                                                model, num_labels, postfix, main_folder_path,
                                                                eval_method, gpu0, useGPU,  patch_size = patch_pred_size, test_augm = test_augm, extra_patch = extra_patch)

                result_iou = METRICS.metricEval('iou', out, gt, num_labels)
                result_dice = METRICS.metricEval('dice', out, gt, num_labels)
                result_recall = METRICS.metricEval('recall', out, gt, num_labels)
                result_precision = METRICS.metricEval('precision', out, gt, num_labels)

                r_list_iou.append(result_iou)
                r_list_dice.append(result_dice)
                r_list_recall.append(result_recall)
                r_list_precision.append(result_precision)
                counter += 1
                print "Model Iter {:5d} Progress: {:4d}/{:4d} iou {:1.4f} dice {:1.4f} recall {:1.4f} precision {:1.4f}  \r".format(iter * 1000, counter, len(img_list), result_iou, result_dice, result_recall, result_precision),
                sys.stdout.flush()
            avg_iou = np.sum(np.asarray(r_list_iou))/len(r_list_iou)
            avg_dice = np.sum(np.asarray(r_list_dice))/len(r_list_dice)
            avg_recall = np.sum(np.asarray(r_list_recall))/len(r_list_recall)
            avg_precision = np.sum(np.asarray(r_list_precision))/len(r_list_precision)
            results_file.write('Iterations: {:5d} iou: {:1.4f} dice: {:1.4f} recall: {:1.4f} precision: {:1.4f} \n'.format(iter*1000, avg_iou, avg_dice, avg_recall, avg_precision))
            print('Done!')
        results_file.close()

def evalModelPriv(model):
    img_list = open(eval_list).readlines()
    if test_mode:
        if models_path == 'None':
            print('Insert model path if you are testing this model')
            sys.exit()
        model = loadSnapshot(model, models_path)

        for img_str in img_list:
            img_str = img_str.rstrip()
            img, gif, out1, gt, out2, affine = EFP.predict(os.path.join(main_folder_path, img_str), 
                                                        model, num_labels, num_labels2, postfix, main_folder_path, eval_method, 
                                                        gpu0, useGPU, patch_size = patch_pred_size, test_augm = test_augm, extra_patch = extra_patch)          
            #save prediction
            save_path = os.path.join('temp_preds', 'pred_' + img_str.split('/')[-3] + '_s' + str(gt.shape[0]) + '.nii.gz')
            PP.saveScan(out2, affine, save_path)
    else:
        if single_eval:
            r = range(1)
        else:
            r = range(iter_low, iter_high, iter_step)
        for iter in r:
            if single_eval:
                model = loadSnapshot(model, models_path)
            else:
                model = loadSnapshot(model, os.path.join(snapshots_path, models_path + '_' + str(iter*1000) + '.pth'))
            counter = 0

            r_list_iou_main = []
            r_list_dice_main = []
            r_list_recall_main = []
            r_list_precision_main = []

            r_list_iou_sec = []

            v = 0
            v_priv = 0

            for img_str in img_list:
                img_str = img_str.rstrip()
                img, gt1, out1, gt2, out2, _ = EFP.predict(os.path.join(main_folder_path, img_str),
                                                                model, num_labels, num_labels2, postfix, main_folder_path, 
                                                                eval_method, gpu0, useGPU,  patch_size = patch_pred_size, test_augm = test_augm, extra_patch = extra_patch)

                result_iou_main = METRICS.metricEval('iou', out2, gt2, num_labels)
                result_dice_main = METRICS.metricEval('dice', out2, gt2, num_labels)
                result_recall_main = METRICS.metricEval('recall', out2, gt2, num_labels)
                result_precision_main = METRICS.metricEval('precision', out2, gt2, num_labels)
                result_iou_sec = METRICS.metricEval('iou', out1, gt1, num_labels2)

                r_list_iou_main.append(result_iou_main)
                r_list_dice_main.append(result_dice_main)
                r_list_recall_main.append(result_recall_main)
                r_list_precision_main.append(result_precision_main)
                r_list_iou_sec.append(result_iou_sec)
                counter += 1
                print "Model Iter | {:5d} | Progress: | {:4d}/{:4d} | Last result {:1.4f}    \r".format(iter * 1000, counter, len(img_list), result_iou_main),
                sys.stdout.flush()
            avg_iou = np.sum(np.asarray(r_list_iou_main))/len(r_list_iou_main)
            avg_dice = np.sum(np.asarray(r_list_dice_main))/len(r_list_dice_main)
            avg_recall = np.sum(np.asarray(r_list_recall_main))/len(r_list_recall_main)
            avg_precision = np.sum(np.asarray(r_list_precision_main))/len(r_list_precision_main)
            avg_iou_sec =  np.sum(np.asarray(r_list_iou_sec))/len(r_list_iou_sec)
            results_file.write('Iterations: {:5d} iou: {:1.4f} dice: {:1.4f} recall: {:1.4f} precision: {:1.4f} iou_secondary: {:1.4f} \n'.format(iter*1000, avg_iou, avg_dice, avg_recall, avg_precision, avg_iou_sec))
        print('Done!')
        results_file.close()

def loadSnapshot(model, path):
    if useGPU:
        #loading on GPU when model was saved on GPU
        saved_state_dict = torch.load(path)
    else:
        #loading on CPU when model was saved on GPU
        saved_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state_dict)
    return model

if __name__ == "__main__":
    model, with_priv = modelInit()
    if with_priv:
        evalModelPriv(model)
    else:
        evalModel(model)