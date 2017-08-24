import sys
sys.path.append('architectures/deeplab_3D/')
sys.path.append('architectures/unet_3D/')
sys.path.append('architectures/hrnet_3D/')
sys.path.append('architectures/experiment_nets_3D/')
sys.path.append('utils/')

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

import nibabel as nib
import evalF as EF
import evalFP as EFP


docstr = """Write something here

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --archId=<int>              Architecture to run, 0 is DeepLab 3D, 1 is U-net3D, 2 is HRNet [default: 2]
    --trainMethod=<int>         0 is full image, 1 is by patches (random), 2 is by patches (center pixel) [default: 1]
    --lossFunction=<str>        Loss function name. 'dice' option available [default: dice]
    --imgSize=<str>             Image size [default: 200x200x100]
    --mainFolderPath=<str>      Main folder path [default: ../Data/MS2017b/]
    --patchSize=<int>           Size of the patch [default: 60]
    --patchSizeStage0=<int>     Size of the patch at stage 0 [default: 41]
    --namePostfix=<str>         Postfix of flair. i.e. to use FLAIR_s postfix is _s. This also determines the train file [default: _200x200x100orig]
    --modelPath=<str>           Path of model to continue training on [default: none]
    --NoLabels=<int>            The number of different labels in training data, including background [default: 2]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --maxIterStage0=<int>       Maximum number of iterations for stage 0 training [default: -1]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 1]
    --lr=<float>                Learning Rate [default: 0.0001]
    --gpu0=<int>                GPU number [default: 0]
    --useGPU=<int>              Use GPU or not [default: 0]
    --experiment=<str>          Specify experiment instead to run. e.g. 1x1x1x1x1x1_1_0 means 1 dilations all 6 blocks, with priv, no ASPP [default: None]
"""
args = docopt(docstr, version='v0.1')
print(args)

arch_id = int(args['--archId'])
train_method = int(args['--trainMethod'])
loss_name = args['--lossFunction']
img_dims = np.array(args['--imgSize'].split('x'), dtype=np.int64)
main_folder_path = args['--mainFolderPath']
patch_size = int(args['--patchSize'])

postfix = args['--namePostfix']
model_path = args['--modelPath']
num_labels = int(args['--NoLabels'])
max_iter = int(args['--maxIter']) 

iter_size = int(args['--iterSize']) 
base_lr = float(args['--lr'])
experiment = str(args['--experiment'])
gpu0 = int(args['--gpu0'])
useGPU = int(args['--useGPU'])
batch_size = 1
#img_dims = [197, 233, 189]
list_path = main_folder_path + 'train' + postfix + '.txt'
print('READING from ', list_path)
img_type_path = 'pre/FLAIR' + postfix + '.nii.gz'
gt_type_path = 'wmh' + postfix + '.nii.gz'


patch_size_stage0 = int(args['--patchSizeStage0'])
max_iter_stage0 = int(args['--maxIterStage0'])

iter_low = 1
iter_high = max_iter + 1

if model_path != 'none':
    iter_low = int(model_path.split('iter_')[-1].replace('.pth','')) + 1
    if iter_low >= iter_high:
        print('Model already at ' + str(iter_low) + ' iterations. Change max iter size')
        sys.exit()

num_labels2 = 209
#change to 0 to enable stage 0 patch learning

if num_labels == 2:
    onlyLesions = True
else:
    onlyLesions = False

if useGPU:
    cudnn.enabled = True
else:
    cudnn.enabled = False

if experiment != 'None':
    snapshot_prefix = 'EXP3D' + '_' + experiment + '_' + loss_name + '_' + str(train_method)
else:
    if arch_id == 0:
        snapshot_prefix = 'DL3D_' + loss_name + '_' + str(train_method) + '_' + PP.getTime()
    elif arch_id == 1:
        snapshot_prefix = 'UNET3D_' + loss_name + '_' + str(train_method) + '_' + PP.getTime()
    elif arch_id == 2:
        snapshot_prefix = 'HR3D' + loss_name + '_' + str(train_method) + '_' + PP.getTime()
to_center_pixel = False
center_pixel_folder_path, locs_lesion, locs_other = (None, None, None)
if train_method == 2:
    to_center_pixel = True
    if not os.path.exists(os.path.join(main_folder_path, 'centerPixelPatches' + postfix + '_' + str(patch_size))):
        print('Pixel patch folder does not exist')
        sys.exit()
#load few files
img_list = PP.read_file(list_path)

results_folder = 'train_results/'
log_file_path = os.path.join(results_folder, 'logs', snapshot_prefix + '_log.txt')
model_file_path = os.path.join(results_folder, 'models', snapshot_prefix + '_best.pth')

logfile = open(log_file_path, 'w+')
info_run = "arch ID: {:d} | max iters: {:10d} | max iters stage 0 : {:10d} | train method : {} | lr : {}".format(arch_id, max_iter, max_iter_stage0, train_method, base_lr)
logfile.write(info_run + '\n')
logfile.flush()

def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def modelInit():
    isPriv = False
    if arch_id > 10:
        isPriv = True

    if experiment != 'None':
        dilation_arr, isPriv, withASPP = PP.getExperimentInfo(experiment)
        model = exp_net_3D.getExpNet(num_labels, dilation_arr, isPriv, NoLabels2 = num_labels2, withASPP = withASPP)
    elif arch_id == 0:
        model = deeplab_resnet_3D.Res_Deeplab(num_labels)
    elif arch_id == 1:
        model = unet_3D.UNet3D(1, num_labels)
    elif arch_id == 2:
        model = highresnet_3D.getHRNet(num_labels)

    if model_path != 'none':
        if useGPU:
            #loading on GPU when model was saved on GPU
            saved_state_dict = torch.load(model_path)
        else:
            #loading on CPU when model was saved on GPU
            saved_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state_dict)

    model.float()
    model.eval() # use_global_stats = True
    return model, isPriv

def trainModel(model):
    if useGPU:
        model.cuda(gpu0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = base_lr)

    optimizer.zero_grad()
    print(model)
    curr_val = 0
    best_val = 0
    val_change = False
    loss_arr = np.zeros([iter_size])
    loss_arr_i = 0
    stage = 0
    print('---------------')
    print('STAGE ' + str(stage))
    print('---------------')

    for iter in range(iter_low, iter_high):
        if iter > max_iter_stage0 and stage != 1:
            print('---------------')
            print('Stage 1')
            print('---------------')
            stage = 1

        if train_method == 0:
            img_b, label_b, _ = PP.extractImgBatch(batch_size, img_list, img_dims, onlyLesions, 
                                                    main_folder_path = '../Data/MS2017b/')
        elif train_method == 1 or train_method == 2:
            if stage == 0:
                batch_size = 1
                img_b, label_b, _ = PP.extractPatchBatch(batch_size, patch_size_stage0, img_list, onlyLesions, center_pixel = to_center_pixel, main_folder_path = '../Data/MS2017b/', postfix=postfix)
            else:
                batch_size = 1
                img_b, label_b, _ = PP.extractPatchBatch(batch_size, patch_size, img_list, onlyLesions, center_pixel = to_center_pixel, main_folder_path = '../Data/MS2017b/', postfix=postfix)
        else:
            print('Invalid training method format')
            sys.exit()

        if stage == 0:
            img_b, label_b = AUG.augmentPatchLossLess([img_b, label_b])
        img_b, label_b = AUG.augmentPatchLossy([img_b, label_b])
        #img_b, label_b = AUG.augmentPatchLossless(img_b, label_b)
        #img_b is of shape      (batch_num) x 1 x dim1 x dim2 x dim3
        #label_b is of shape    (batch_num) x 1 x dim1 x dim2 x dim3
        #batch_num should be 1 since too memory intensive

        label_b = label_b.astype(np.int64)
        #convert label from (batch_num x 1 x dim1 x dim2 x dim3)
        #               to  ((batch_numxdim1*dim2*dim3) x 3) (one hot)
        temp = label_b.reshape([-1])
        label_b = np.zeros([temp.size, num_labels])
        label_b[np.arange(temp.size),temp] = 1
        label_b = torch.from_numpy(label_b).float()

        imgs = torch.from_numpy(img_b).float()

        if useGPU:
            imgs, label_b = Variable(imgs).cuda(gpu0), Variable(label_b).cuda(gpu0)
        else:
            imgs, label_b = Variable(imgs), Variable(label_b)

        #---------------------------------------------
        #out size is      (1, 3, dim1, dim2, dim3)
        #---------------------------------------------
        out = model(imgs)
        out = out.permute(0,2,3,4,1).contiguous()
        out = out.view(-1, num_labels)
        #---------------------------------------------
        #out size is      (1 * dim1 * dim2 * dim3, 3)
        #---------------------------------------------

        #loss function
        m = nn.Softmax()
        loss = lossF.simple_dice_loss3D(m(out), label_b)

        loss /= iter_size
        loss.backward()

        loss_val = loss.data.cpu().numpy()
        loss_arr[loss_arr_i] = loss_val
        loss_arr_i = (loss_arr_i + 1) % iter_size

        if iter % 1 == 0:
            if val_change:
                print "iter = {:6d}/{:6d}       Loss: {:1.6f}       Val Score: {:1.6f}     \r".format(iter-1, max_iter, float(loss_val)*iter_size, curr_val),
                sys.stdout.flush()
                print ""
                val_change = False
            print "iter = {:6d}/{:6d}       Loss: {:1.6f}       Val Score: {:1.6f}     \r".format(iter, max_iter, float(loss_val)*iter_size, curr_val),
            sys.stdout.flush()
        if iter % 1000 == 0:
            val_change = True
            curr_val = EF.evalModelX(model, num_labels, postfix, main_folder_path, (train_method != 0), gpu0, useGPU, eval_metric = 'iou', patch_size = patch_size, extra_patch = 5)
            if curr_val > best_val:
                best_val = curr_val
                print('\nSaving better model...')
                torch.save(model.state_dict(), model_file_path)
            logfile.write("iter = {:6d}/{:6d}       Loss: {:1.6f}       Val Score: {:1.6f}     \n".format(iter, max_iter, np.sum(loss_arr), curr_val))
            logfile.flush()
        if iter % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        del out, loss

def setupGIFVar(gif_b):
    gif_b = gif_b.astype(np.int64)
    gif_b = gif_b.reshape([-1])
    gif_b = torch.from_numpy(gif_b).long()

    if useGPU:
        gif_b = Variable(gif_b).cuda(gpu0)
    else:
        gif_b = Variable(gif_b)
    return gif_b

def trainModelPriv(model):
    if useGPU:
        model.cuda(gpu0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = base_lr)
    optimizer.zero_grad()
    print(model)
    curr_val1 = 0
    curr_val2 = 0
    best_val2 = 0
    val_change = False
    loss_arr1 = np.zeros([iter_size])
    loss_arr2 = np.zeros([iter_size])
    loss_arr_i = 0

    stage = 0
    print('---------------')
    print('STAGE ' + str(stage))
    print('---------------')

    for iter in range(iter_low, iter_high):
        if iter > max_iter_stage0 and stage != 1:
            print('---------------')
            print('Stage 1')
            print('---------------')
            stage = 1

        if train_method == 0:
            img_b, label_b, gif_b = PP.extractImgBatch(batch_size, img_list, img_dims, onlyLesions, 
                                                    main_folder_path = '../Data/MS2017b/', with_priv = True)
        elif train_method == 1 or train_method == 2:
            if stage == 0:
                batch_size = 5
                img_b, label_b, gif_b = PP.extractPatchBatch(batch_size, patch_size_stage0, img_list, onlyLesions,
                                                            center_pixel = to_center_pixel, 
                                                            main_folder_path = '../Data/MS2017b/', 
                                                            postfix=postfix, with_priv= True)
            else:
                batch_size = 1
                img_b, label_b, gif_b = PP.extractPatchBatch(batch_size, patch_size, img_list, onlyLesions, 
                                                    center_pixel = to_center_pixel, 
                                                    main_folder_path = '../Data/MS2017b/', 
                                                    postfix=postfix, with_priv= True)
        else:
            print('Invalid training method format')
            sys.exit()

        img_b, label_b, gif_b = AUG.augmentPatchLossy([img_b, label_b, gif_b])

        #img_b is of shape      (batch_num) x 1 x dim1 x dim2 x dim3
        #label_b is of shape    (batch_num) x 1 x dim1 x dim2 x dim3

        label_b = label_b.astype(np.int64)

        #convert label from (batch_num x 1 x dim1 x dim2 x dim3)
        #               to  ((batch_numxdim1*dim2*dim3) x 3) (one hot)
        temp = label_b.reshape([-1])
        label_b = np.zeros([temp.size, num_labels])
        label_b[np.arange(temp.size),temp] = 1
        label_b = torch.from_numpy(label_b).float()

        imgs = torch.from_numpy(img_b).float()

        if useGPU:
            imgs, label_b = Variable(imgs).cuda(gpu0), Variable(label_b).cuda(gpu0)
        else:
            imgs, label_b = Variable(imgs), Variable(label_b)

        gif_b = setupGIFVar(gif_b)

        #---------------------------------------------
        #out size is      (1, 3, dim1, dim2, dim3)
        #---------------------------------------------
        #out1 is extra info
        out1, out2 = model(imgs)

        out1 = out1.permute(0,2,3,4,1).contiguous()
        out1 = out1.view(-1, num_labels2)

        out2 = out2.permute(0,2,3,4,1).contiguous()
        out2 = out2.view(-1, num_labels)
        #---------------------------------------------
        #out size is      (1 * dim1 * dim2 * dim3, 3)
        #---------------------------------------------
        m2 = nn.Softmax()
        loss2 = lossF.simple_dice_loss3D(m2(out2), label_b)
        m1 = nn.LogSoftmax()
        loss1 = F.nll_loss(m1(out1), gif_b)

        loss1 /= iter_size
        loss2 /= iter_size

        torch.autograd.backward([loss1, loss2])

        loss_val1 = float(loss1.data.cpu().numpy())
        loss_arr1[loss_arr_i] = loss_val1

        loss_val2 = float(loss2.data.cpu().numpy())
        loss_arr2[loss_arr_i] = loss_val2

        loss_arr_i = (loss_arr_i + 1) % iter_size

        if iter % 1 == 0:
            if val_change:
                print "iter = {:6d}/{:6d}       Loss_main: {:1.6f}    Loss_secondary: {:1.6f}       Val Score: {:1.6f}      Val Score secondary: {:1.6f}     \r".format(iter-1, max_iter, loss_val2*iter_size, loss_val1*iter_size, curr_val2, curr_val1),
                sys.stdout.flush()
                print ""
                val_change = False
            print "iter = {:6d}/{:6d}       Loss_main: {:1.6f}      Loss_secondary: {:1.6f}       Val Score main: {:1.6f}      Val Score secondary: {:1.6f}     \r".format(iter, max_iter, loss_val2*iter_size, loss_val1*iter_size, curr_val2, curr_val1),
            sys.stdout.flush()
        if iter % 2000 == 0:
            val_change = True
            curr_val1, curr_val2 = EFP.evalModelX(model, num_labels, num_labels2, postfix, main_folder_path, (train_method != 0), gpu0, useGPU, eval_metric = 'iou', patch_size = patch_size, extra_patch = 5, priv_eval = True)
            if curr_val2 > best_val2:
                best_val2 = curr_val2
                torch.save(model.state_dict(), model_file_path)
                print('\nSaving better model...')
            logfile.write("iter = {:6d}/{:6d}       Loss_main: {:1.6f}      Loss_secondary: {:1.6f}       Val Score main: {:1.6f}      Val Score secondary: {:1.6f}  \n".format(iter, max_iter, np.sum(loss_arr2), np.sum(loss_arr1), curr_val2, curr_val1))
            logfile.flush()
        if iter % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        del out1, out2, loss1, loss2

if __name__ == "__main__":
    model, with_priv = modelInit()
    if with_priv:
        trainModelPriv(model)
    else:
        trainModel(model)
    logfile.close()