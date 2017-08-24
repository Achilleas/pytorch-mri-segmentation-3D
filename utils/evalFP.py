import sys
import os

import evalMetrics as METRICS
import PP
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import augmentations as AUG

#---------------------------------------------
#Evaluation functions for PrivCNNs
#---------------------------------------------

def evalModelX(model, num_labels, num_labels2, postfix, main_folder_path, eval_method, gpu0, useGPU,
                    patch_size = 70, eval_metric = 'iou', test_augm = False, extra_patch = 30, priv_eval = True):
    eval_list = main_folder_path + 'val' + postfix + '.txt'
    img_list = open(eval_list).readlines()
    v = 0
    v_priv = 0
    for img_str in img_list:
        img_str = img_str.rstrip()
        _, gt1, out1, gt2, out2, _ = predict(os.path.join(main_folder_path, img_str), model, num_labels, num_labels2, 
                                                    postfix, main_folder_path, eval_method, gpu0, useGPU, patch_size=patch_size, 
                                                    test_augm = test_augm, extra_patch = extra_patch, priv_eval = priv_eval)
        v += METRICS.metricEval(eval_metric, out2, gt2, num_labels)
        v_priv += METRICS.metricEval(eval_metric, out1, gt1, num_labels2)
    return v_priv / len(img_list), v / len(img_list)


def testPredict(img, model, num_labels, num_labels2, eval_method, gpu0, useGPU, stride = 50, patch_size = 70, test_augm = True, extra_patch = 30, get_soft = False):
    if eval_method == 0:
        if useGPU:
            _, out = model(Variable(torch.from_numpy(img).float(),volatile = True).cuda(gpu0))
        else:
            _, out = model(Variable(torch.from_numpy(img).float(),volatile = True))
        out = out.data[0].cpu().numpy()
    elif eval_method == 1:
            _, out = predictByPatches(img, model, num_labels, num_labels2, useGPU, gpu0, 
                    stride = stride, patch_size = patch_size, 
                    test_augm = test_augm, extra_patch = extra_patch, priv_eval = False)
    out = out.squeeze()
    if get_soft:
        return out
    #take argmax to get predictions
    out = np.argmax(out, axis = 0)
    #remove batch and label dimension
    out = out.squeeze()
    return out

#returns the image as numpy, the ground truth and the prediction given model and input path
#affine = True, returns the affine transformation from loading the scan
def predict(img_path, model, num_labels, num_labels2, postfix, main_folder_path, eval_method, gpu0, useGPU, 
                        stride = 50, patch_size = 70, test_augm = True, extra_patch = 30, priv_eval = True):
    #read image
    img = PP.numpyFromScan(img_path)
    #read wmh
    gt_path = img_path.replace('slices', 'gt_slices').replace('FLAIR', 'wmh').replace('/pre','')
    gt, affine = PP.numpyFromScan(gt_path, get_affine = True, makebin = (num_labels == 2))

    gif_path = img_path.replace('scans', 'gifs').replace('FLAIR', 'parcellation').replace('/pre','')
    gif = PP.numpyFromScan(gif_path)

    img = img.transpose((3,0,1,2))
    img = img[np.newaxis, :]
    gt = gt.transpose((3,0,1,2))
    gif = gif.transpose((3,0,1,2))

    if eval_method == 0:
        if useGPU:
            out1_v, out2_v = model(Variable(torch.from_numpy(img).float(),volatile=True).cuda(gpu0))
        else:
            out1_v, out2_v = model(Variable(torch.from_numpy(img).float(),volatile=True))
        out1 = out1_v.data[0].cpu().numpy()
        out2 = out2_v.data[0].cpu().numpy()
        del out1_v, out2_v
    elif eval_method == 1:
            out1, out2 = predictByPatches(img, model, num_labels, num_labels2, useGPU, gpu0, 
                    stride = stride, test_augm = test_augm, patch_size = patch_size, 
                    extra_patch = extra_patch, priv_eval = priv_eval)
    out1 = out1.squeeze()
    out1 = np.argmax(out1, axis = 0)
    out1 = out1.squeeze()

    out2 = out2.squeeze()
    out2 = np.argmax(out2, axis = 0)
    out2 = out2.squeeze()

    #remove batch and label dimension
    img = img.squeeze()

    return img, gif, out1, gt, out2, affine

def predictByPatches(img, model, num_labels, num_labels2, useGPU, gpu0, patch_size = 70, test_augm = False, stride = 50, extra_pad = 0, extra_patch = 30, priv_eval = True):    
    batch_num, num_channels, dim1, dim2, dim3 = img.shape
    p_size = patch_size
    #add padding to each dim s.t. % stride = 0
    dim1_pad = stride - ((dim1-p_size) % stride)
    dim2_pad = stride - ((dim2-p_size) % stride)
    dim3_pad = stride - ((dim3-p_size) % stride)

    x_1_off, x_2_off = int(round(dim1_pad/2.0)), dim1_pad//2
    y_1_off, y_2_off = int(round(dim2_pad/2.0)), dim2_pad//2
    z_1_off, z_2_off = int(round(dim3_pad/2.0)), dim3_pad//2

    img = np.lib.pad(img, ((0,0),(0,0), (x_1_off, x_2_off), (y_1_off, y_2_off), (z_1_off, z_2_off)), mode='minimum')
    _, _, padded_dim1, padded_dim2, padded_dim3 = img.shape

    out2_shape = (img.shape[0], num_labels, img.shape[2], img.shape[3], img.shape[4])
    out1_shape = (img.shape[0], num_labels2, img.shape[2], img.shape[3], img.shape[4])

    out1_total = np.zeros(out1_shape, dtype=np.float16)
    out1_counter = np.zeros(out1_shape, dtype=np.int8)
    out2_total = np.zeros(out2_shape)
    out2_counter = np.zeros(out2_shape)
    
    extra_p = extra_patch / 2

    for i in range(0, padded_dim1 - p_size + 1, stride):
        for j in range(0, padded_dim2 - p_size + 1, stride):
            for k in range(0, padded_dim3 - p_size + 1, stride):
                if extra_p != 0:
                    i_l, i_r = getExtraPatchOffsets(i, 0, padded_dim1 - p_size, extra_p)
                    j_l, j_r = getExtraPatchOffsets(j, 0, padded_dim2 - p_size, extra_p)
                    k_l, k_r = getExtraPatchOffsets(k, 0, padded_dim3 - p_size, extra_p)

                    img_patch = img[:,:, (i-i_l):(i+p_size+i_r),(j-j_l):(j+p_size+j_r),(k-k_l):(k+p_size+k_r)]

                    out1_np, out2_np = getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = extra_pad, test_augm = test_augm)
                    out1_np = removePatchOffset(out1_np, i_l, i_r, j_l, j_r, k_l, k_r)
                    out2_np = removePatchOffset(out2_np, i_l, i_r, j_l, j_r, k_l, k_r)

                    if priv_eval:
                        out1_total[:,:, i:i+p_size,j:j+p_size,k:k+p_size] += out1_np
                        out1_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1
                    out2_total[:,:, i:i+p_size,j:j+p_size,k:k+p_size] += out2_np
                    out2_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1

                else:
                    img_patch = img[:, :, i:i+p_size, j:j+p_size, k:k+p_size]
                    #make a prediction on this image patch, adding extra padding during prediction and augmenting
                    #the result is of the same shape and size as the original img patch
                    out1_np, out2_np = getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = extra_pad, test_augm = test_augm)
                    #too memory intensive
                    if priv_eval:
                        out1_total[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += out1_np.astype(np.float16)
                        out1_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1
                    out2_total[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += out2_np
                    out2_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1
    if priv_eval:
        out1_total = out1_total / out1_counter
    out2_total = out2_total / out2_counter
    #remove padding from predictions
    out1_total = out1_total[:, :, x_1_off:-x_2_off, y_1_off:-y_2_off, z_1_off:-z_2_off]
    out2_total = out2_total[:, :, x_1_off:-x_2_off, y_1_off:-y_2_off, z_1_off:-z_2_off]
    return out1_total, out2_total

def getExtraPatchOffsets(v, low_bound, upper_bound, extra_p):
    v_left = 0
    v_right = 0
    if v - extra_p > low_bound:
        v_left = extra_p
    if v + extra_p < upper_bound:
        v_right = extra_p  
    return v_left, v_right

#list of tuple [(i_l, i_r), (j_l, j_r)]
def removePatchOffset(np_arr, i_l, i_r, j_l, j_r, k_l, k_r):
    bn, c, s_i, s_j, s_k = np_arr.shape
    return np_arr[:,:,(i_l):(s_i-i_r), (j_l):(s_j-j_r), (k_l):(s_k-k_r)]


def getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = 10, test_augm = False):
    pd = extra_pad/2
    padding = ((0,0), (0,0), (pd, pd), (pd, pd), (pd,pd))
    img_patch = np.pad(img_patch, padding, 'constant')

    num_augm = 1
    if test_augm:
        num_augm = 3

    out1_np_total = None
    out2_np_total = None
    for i in range(num_augm):
        img_patch_cp = np.copy(img_patch)
        if test_augm and i != 0:
            #apply augmentation
            rot_x, rot_y, rot_z = AUG.getRotationVal([10,10,10])
            zoom_val = AUG.getScalingVal(0.9, 1.1)

            img_patch_cp = AUG.applyScale([img_patch_cp], zoom_val, [3])[0]
            img_patch_cp = AUG.applyRotation([img_patch_cp], [rot_x, rot_y, rot_z], [3])[0]
        if useGPU:
            out1, out2 = model(Variable(torch.from_numpy(img_patch_cp).float(),volatile=True).cuda(gpu0))
        else:
            out1, out2 = model(Variable(torch.from_numpy(img_patch_cp).float(),volatile=True))

        out1_np = out1.data[0].cpu().numpy()
        out2_np = out2.data[0].cpu().numpy()
        del out1, out2
        #output is (1 x 3 x dim1 x dim2 x dim3)
        out1_np = out1_np[np.newaxis,:]
        out2_np = out2_np[np.newaxis,:]

        if test_augm and i != 0:
            temp2 = np.copy(out2_np)
            out2_np = None
            rev_zoom_i = float(img_patch.shape[2]) / img_patch_cp.shape[2]
            rev_zoom_j = float(img_patch.shape[3]) / img_patch_cp.shape[3]
            rev_zoom_k = float(img_patch.shape[4]) / img_patch_cp.shape[4]

            for j in range(temp2.shape[1]):
                r2 = AUG.applyRotation([temp2[:,j:j+1,:,:,:]], [-rot_x, -rot_y, -rot_z], [3])[0]
                r2 = AUG.applyScale([r2], [rev_zoom_i,rev_zoom_j,rev_zoom_k], [3])[0]
                if not isinstance(out2_np, np.ndarray):
                    out2_np = np.zeros([1, temp2.shape[1], r2.shape[2], r2.shape[3], r2.shape[4]])
                out2_np[:, j,:,:,:] = r2

        out2_np = numpySoftmax(out2_np, 1)

        nb, c, n_i, n_j, n_k = out2_np.shape

        if not isinstance(out1_np_total, np.ndarray):
            out1_np_total = out1_np[:,:,(pd):(n_i-pd),(pd):(n_j-pd),(pd):(n_k-pd)]
            out2_np_total = out2_np[:,:,(pd):(n_i-pd),(pd):(n_j-pd),(pd):(n_k-pd)]
        else:
            out2_np_total += out2_np[:,:,(pd):(n_i-pd),(pd):(n_j-pd),(pd):(n_k-pd)]
    gc.collect()
    return (out1_np_total), (out2_np_total / num_augm)

def numpySoftmax(x, axis_):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=axis_) + 0.00001)
