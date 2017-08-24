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
#Evaluation functions
#---------------------------------------------

def evalModelX(model, num_labels, postfix, main_folder_path, eval_method, gpu0, useGPU,
                    patch_size = 70, eval_metric = 'iou', test_augm = False, extra_patch = 30):
    eval_list = main_folder_path + 'val' + postfix + '.txt'
    img_list = open(eval_list).readlines()
    v = 0
    v_priv = 0
    for img_str in img_list:
        img_str = img_str.rstrip()
        _, gt, out, _ = predict(os.path.join(main_folder_path, img_str), model, num_labels, postfix,
                                            main_folder_path, eval_method, gpu0, useGPU, patch_size=patch_size,
                                                test_augm = test_augm, extra_patch = extra_patch)
        curr_eval = METRICS.metricEval(eval_metric, out, gt, num_labels)
        v+=curr_eval
    return v / len(img_list)

def testPredict(img, model, num_labels, eval_method, gpu0, useGPU, stride= 50, patch_size = 70, test_augm = True, extra_patch = 30, get_soft = False):
    if eval_method == 0:
        if useGPU:
            out = model(Variable(torch.from_numpy(img).float(),volatile = True).cuda(gpu0))
        else:
            out = model(Variable(torch.from_numpy(img).float(),volatile = True))
        out = out.data[0].cpu().numpy()
    elif eval_method == 1:
            out = predictByPatches(img, model, num_labels, useGPU, gpu0, 
                stride = stride, patch_size = patch_size, 
                test_augm = test_augm, extra_patch = extra_patch)
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
def predict(img_path, model, num_labels, postfix, main_folder_path, eval_method, gpu0, useGPU, 
                        stride = 50, patch_size = 70, test_augm = True, extra_patch = 30):  
    #read image
    img = PP.numpyFromScan(img_path)
    #read wmh
    gt_path = img_path.replace('slices', 'gt_slices').replace('FLAIR', 'wmh').replace('/pre','')
    gt, affine = PP.numpyFromScan(gt_path, get_affine = True, makebin = (num_labels == 2))

    img = img.transpose((3,0,1,2))
    img = img[np.newaxis, :]
    gt = gt.transpose((3,0,1,2))

    if eval_method == 0:
        if useGPU:
            out_v = model(Variable(torch.from_numpy(img).float(),volatile = True).cuda(gpu0))
        else:
            out_v = model(Variable(torch.from_numpy(img).float(),volatile = True))
        out = out_v.data[0].cpu().numpy()
        #FIX?
        del out_v
        out_v = Variable(torch.from_numpy(np.array([1])).float())
        out_v = Variable(torch.from_numpy(np.array([1])).float())
    elif eval_method == 1:
            out = predictByPatches(img, model, num_labels, useGPU, gpu0, stride = stride, patch_size = patch_size, test_augm = test_augm, extra_patch = extra_patch)
    out = out.squeeze()
    #take argmax to get predictions
    out = np.argmax(out, axis = 0)
    #remove batch and label dimension
    img = img.squeeze()
    out = out.squeeze()
    gt = gt.squeeze()

    return img, gt, out, affine

def predictByPatches(img, model, num_labels, useGPU, gpu0, patch_size = 70, test_augm = False, stride = 50, extra_pad = 0, extra_patch = 30):    
    batch_num, num_channels, dim1, dim2, dim3 = img.shape
    p_size = patch_size
    #add padding to each dim s.t. % stride = 0
    dim1_pad = (stride - ((dim1-p_size) % stride)) % stride
    dim2_pad = (stride - ((dim2-p_size) % stride)) % stride
    dim3_pad = (stride - ((dim3-p_size) % stride)) % stride

    x_1_off, x_2_off = int(round(dim1_pad/2.0)), dim1_pad//2
    y_1_off, y_2_off = int(round(dim2_pad/2.0)), dim2_pad//2
    z_1_off, z_2_off = int(round(dim3_pad/2.0)), dim3_pad//2

    img = np.lib.pad(img, ((0,0),(0,0), (x_1_off, x_2_off), (y_1_off, y_2_off), (z_1_off, z_2_off)), mode='minimum')
    _, _, padded_dim1, padded_dim2, padded_dim3 = img.shape

    out_shape = (img.shape[0], num_labels, img.shape[2], img.shape[3], img.shape[4])
    out_total = np.zeros(out_shape)
    out_counter = np.zeros(out_shape)

    extra_p = extra_patch / 2
    for i in range(0, padded_dim1 - p_size + 1, stride):
        for j in range(0, padded_dim2 - p_size + 1, stride):
            for k in range(0, padded_dim3 - p_size + 1, stride):
                if extra_p != 0:
                    i_l, i_r = getExtraPatchOffsets(i, 0, padded_dim1 - p_size, extra_p)
                    j_l, j_r = getExtraPatchOffsets(j, 0, padded_dim2 - p_size, extra_p)
                    k_l, k_r = getExtraPatchOffsets(k, 0, padded_dim3 - p_size, extra_p)

                    img_patch = img[:,:, (i-i_l):(i+p_size+i_r),(j-j_l):(j+p_size+j_r),(k-k_l):(k+p_size+k_r)]
                    out_np = getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = extra_pad, test_augm = test_augm)
                    out_np = removePatchOffset(out_np, i_l, i_r, j_l, j_r, k_l, k_r)
                    out_total[:,:, i:i+p_size,j:j+p_size,k:k+p_size] += out_np
                    out_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1
                else:
                    img_patch = img[:, :, i:i+p_size, j:j+p_size, k:k+p_size]
                    #make a prediction on this image patch, adding extra padding during prediction and augmenting
                    #the result is of the same shape and size as the original img patch
                    out_np = getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = extra_pad, test_augm = test_augm)

                    out_total[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += out_np
                    out_counter[:, :, i:i+p_size, j:j+p_size, k:k+p_size] += 1
    out_total = out_total / out_counter
    #remove padding from predictions
    nb, c, i_size, j_size, k_size = out_total.shape
    out_total = out_total[:, :, x_1_off:i_size-x_2_off, y_1_off:j_size-y_2_off, z_1_off:k_size-z_2_off]

    return out_total

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

def getPatchPrediction(img_patch, model, useGPU, gpu0, extra_pad = 0, test_augm = False):
    pd = extra_pad/2
    padding = ((0,0), (0,0), (pd, pd), (pd, pd), (pd,pd))
    img_patch = np.pad(img_patch, padding, 'constant')

    num_augm = 1
    if test_augm:
        num_augm = 3

    out_np_total = None
    for i in range(num_augm):
        img_patch_cp = np.copy(img_patch)
        #AUGMENT IMAGE
        if test_augm and i != 0:
            pass
            #apply augmentation
            rot_x, rot_y, rot_z = AUG.getRotationVal([10,10,10])
            zoom_val = AUG.getScalingVal(0.8, 1.1)

            img_patch_cp = AUG.applyScale([img_patch_cp], zoom_val, [3])[0]
            img_patch_cp = AUG.applyRotation([img_patch_cp], [rot_x, rot_y, rot_z], [3])[0]

        #MAKE PREDICTION
        if useGPU:
            out = model(Variable(torch.from_numpy(img_patch_cp).float(),volatile = True).cuda(gpu0))
        else:
            out = model(Variable(torch.from_numpy(img_patch_cp).float(),volatile = True))
        out_np = out.data[0].cpu().numpy()
        #output is (1 x 3 x dim1 x dim2 x dim3)
        out_np = out_np[np.newaxis,:]
        if test_augm and i != 0:
            temp = np.copy(out_np)
            out_np = None
            #reverse augmentation on predictions
            rev_zoom_i = float(img_patch.shape[2]) / img_patch_cp.shape[2]
            rev_zoom_j = float(img_patch.shape[3]) / img_patch_cp.shape[3]
            rev_zoom_k = float(img_patch.shape[4]) / img_patch_cp.shape[4]

            for j in range(temp.shape[1]):
                r = AUG.applyRotation([temp[:,j:j+1,:,:,:]], [-rot_x, -rot_y, -rot_z], [3])[0]
                r = AUG.applyScale(r, [rev_zoom_i,rev_zoom_j,rev_zoom_k], [3])[0]

                if not isinstance(out_np, np.ndarray):
                    out_np = np.zeros([1, temp.shape[1], r.shape[2], r.shape[3], r.shape[4]])
                out_np[:, j,:,:,:] = r
        out_np = numpySoftmax(out_np, 1)
        if not isinstance(out_np_total, np.ndarray):
            if pd == 0:
                out_np_total = out_np
            else:
                out_np_total = out_np[:,:,pd:-pd, pd:-pd, pd:-pd]
        else:
            if pd ==0:
                out_np_total += out_np
            else:
                out_np_total += out_np[:,:,pd:-pd, pd:-pd, pd:-pd]

    return out_np_total / num_augm



def numpySoftmax(x, axis_):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=axis_) + 0.00001)
