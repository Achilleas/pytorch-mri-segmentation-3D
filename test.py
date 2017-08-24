import numpy as np
import sys
import os
import glob
import nibabel as nib
import torch
#docker
#fpx = '/wmhseg_code/'
#inputDir = '/input'
#outputDir = '/output'

#local
fpx = './'
inputDir = '/input'
outputDir = '/output'

#PARAMS
useGPU = 0
gpu0 = 0
patch_size = 60
extra_patch = 5
model_paths = [fpx + 'EXPNETXXX.pth']
weights = [1]

sys.path.append(fpx + '/utils/')
sys.path.append(fpx + 'architectures/deeplab_3D/')
sys.path.append(fpx + 'architectures/unet_3D/')
sys.path.append(fpx + 'architectures/hrnet_3D/')
sys.path.append(fpx + 'architectures/experiment_nets_3D/')
sys.path.append('utils/')

import deeplab_resnet_3D
import unet_3D
import highresnet_3D
import exp_net_3D

import augmentations as AUG
import normalizations as NORM
import resizeScans as RS
import evalF as EF
import evalFP as EFP
import PP
import torch

#step 1: read image from input folder (pre/)
#step 2: resize image to 200x200x100 + apply normalizations
#step 3: make prediction by patches (with augmentations)
#step 4: save prediction to output folder
#step 5: resize prediction back to original size of image


img_path = os.path.join(inputDir, 'pre', 'FLAIR.nii.gz')
img_path_rs = os.path.join(outputDir, 'FLAIR_rs.nii.gz')

wmh_path_rs = os.path.join(outputDir, 'wmh_rs.nii.gz')
wmh_path = os.path.join(outputDir, 'result.nii.gz')

old_size = PP.numpyFromScan(img_path).shape

new_size = [200,200,100]
num_labels = 2

#convert scan to 200x200x100
RS.convertSize2(img_path, img_path_rs, new_size)
#get the affine value
affine_rs = nib.load(img_path_rs).get_affine()

#normalize using histogram and variance normalization
RS.normalizeScan(img_path_rs, img_path_rs, main_folder_path=fpx)

print(glob.glob('/output/*'))
#read preprocessed img
img, affine = PP.numpyFromScan(img_path_rs, get_affine = True)
img = img.transpose((3,0,1,2))
img = img[np.newaxis, :]

print('Image ready')
print('Loading model')

out = None
for i, model_path in enumerate(model_paths): 
	f_name = model_path.split('/')[-1]
	isPriv = False

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

	if useGPU:
	    saved_state_dict = torch.load(model_path)
	else:
	    saved_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(saved_state_dict)
	model.float()
	model.eval()
	print('Model ready')
	print('Predicting...')
	if not isinstance(out, np.ndarray):
		if isPriv:
			out = EFP.testPredict(img, model, num_labels, 209, 1, gpu0, useGPU, stride = 50, patch_size = 60, test_augm = True, extra_patch = extra_patch, get_soft = True)
		else:
			out = EF.testPredict(img, model, num_labels, 1, gpu0, useGPU, stride = 50, patch_size = 60, test_augm = True, extra_patch = extra_patch, get_soft = True)
	else:
		if isPriv:
			out += EFP.testPredict(img, model, num_labels, 209, 1, gpu0, useGPU, stride = 50, patch_size = 60, test_augm = True, extra_patch = extra_patch, get_soft = True)
		else:
			out += EF.testPredict(img, model, num_labels, 1, gpu0, useGPU, stride = 50, patch_size = 60, test_augm = True, extra_patch = extra_patch, get_soft = True)

out /= float(len(model_paths))
out = np.argmax(out, axis = 0)
#remove batch and label dimension
out = out.squeeze()
print('Prediction complete')
print('Saving...')
#save output
PP.saveScan(out.astype(np.float64), affine_rs, wmh_path_rs)

#resize output to original input size and save (this is our final result)
d = RS.convertSize2(wmh_path_rs, wmh_path, old_size, interpolation = 'nearest')

#read the image and save it with same affine and header as original FLAIR image
print('Saving final wmh file')
orig_flair = nib.load(img_path)
wmh_final = nib.load(wmh_path).get_data()
PP.saveScan(wmh_final, orig_flair.get_affine(), wmh_path, header =orig_flair.header)
print('Done')