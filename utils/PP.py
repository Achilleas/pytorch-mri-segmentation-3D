#Preprocessing file
import os
import numpy as np
import nibabel as nib
import time
import math
import random
import glob
import collections
import sys
import random
from random import randint
import time
import datetime
import augmentations as AUG

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Basic Utils
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getTime():
	return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
	
def printDimensions(img_path = 'pre/FLAIR.nii.gz', segm_path = 'wmh.nii.gz', data_folder = '../Data/MS2017a/'):
	scan_folders = glob.glob(data_folder + 'scans/*')

	for sf in scan_folders:
		file_num = os.path.basename(sf)
		img = nib.load(os.path.join(sf, img_path))
		print(file_num, img.shape)

def extractMeanDataStats(size = [200, 200, 100], 
						postfix = '_200x200x100orig', 
						main_folder_path = '../../Data/MS2017b/', 
						):
	scan_folders = glob.glob(main_folder_path + 'scans/*')
	img_path = 'pre/FLAIR' + postfix + '.nii.gz'
	segm_path = 'wmh' + postfix + '.nii.gz'
	
	shape_ = [len(scan_folders), size[0], size[1], size[2]]
	arr = np.zeros(shape_)

	for i, sf in enumerate(scan_folders):
		arr[i, :,:,:] =  numpyFromScan(os.path.join(sf,img_path)).squeeze()

	arr /= len(scan_folders)

	means = np.mean(arr)
	stds = np.std(arr, axis = 0)

	np.save(main_folder_path + 'extra_data/std' + postfix, stds)
	np.save(main_folder_path + 'extra_data/mean' + postfix, means)

def getExperimentInfo(experiment_str):
    exp_arr = experiment_str.split('_')
    isPriv = bool(int(exp_arr[1]))
    withASPP = bool(int(exp_arr[2]))
    dilations_str = exp_arr[0]
    dilation_arr = [int(i) for i in dilations_str.split('x')]

    return dilation_arr, isPriv, withASPP

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
##GENERAL PREPROCESSING
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#convert .nii.gz file to scan
def numpyFromScan(path, get_affine = False, makebin = False):
	img = nib.load(path)
	img_np = img.get_data()
	#reshape to size1 x size2  - > size1 x size2 x 1
	img_np = np.expand_dims(img_np, axis=len(img_np.shape))
	#img_np = img_np.reshape([img_np.shape[0], img_np.shape[1], 1])
	if makebin:
		img_np[img_np == 2] = 0

	if get_affine:
		return img_np, img.get_affine()
	return img_np

def saveScan(img_np, affine, save_path, header = None):
	if header:
		nft_img = nib.Nifti1Image(img_np, affine, header = header)
	else:
		nft_img = nib.Nifti1Image(img_np, affine)
	nib.save(nft_img, save_path)

#get list of validation/train sets
def splitTrainVal(train_fraction, data_folder = '../Data/MS2017a/'):
	scan_folders = glob.glob(data_folder + 'scans/*')
	num_scans = len(scan_folders)

	indices = np.random.permutation(num_scans)
	train_indices = indices[0:int(num_scans*train_fraction)]
	val_indices = indices[int(num_scans*train_fraction):]

	train_scan_folders = [scan_folders[i] for i in train_indices]
	val_scan_folders = [scan_folders[i] for i in val_indices]

	return train_scan_folders, val_scan_folders

#call this once to split training data
def generateTrainValFile(train_fraction, main_folder = '../Data/MS2017a/', postfix = ''):
	train_folders, val_folders = splitTrainVal(0.8, data_folder=main_folder)

	img_path = '/pre/FLAIR' + postfix + '.nii.gz'
	train_folder_names = [train_folders[i].split(main_folder)[1] + img_path for i in range(len(train_folders))]
	val_folder_names = [val_folders[i].split(main_folder)[1]  + img_path for i in range(len(val_folders))]

	f_train = open(main_folder + 'train' + postfix + '.txt', 'w+')
	f_val = open(main_folder + 'val' + postfix + '.txt', 'w+')

	for fn in train_folder_names:
		f_train.write(fn + '\n')

	for fn in val_folder_names:
		f_val.write(fn + '\n')

	f_train.close()
	f_val.close()

def read_file(path_to_file, pretext = ''):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(pretext + line[:-1])
    return img_list


def generateTestFile(folder):
	pass

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
##END GENERAL PREPROCESSING
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



#func1()
#simpleSplitTrainVal(0.8)
#generateTrainValFile(0.8)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
##SLICES PREPROCESSING
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#generate a slices folder containing all slices
def generateImgSlicesFolder(data_folder = '../Data/MS2017a/scans/'):
	scan_folders = glob.glob(data_folder + '*')

	for sf in scan_folders:
		slice_dir_path = os.path.join(sf, 'slices/')
		if not os.path.exists(slice_dir_path):
			print('Creating directory at:' , slice_dir_path)
			os.makedirs(slice_dir_path)

		img = nib.load(os.path.join(sf, 'pre/FLAIR.nii.gz'))
		img_np = img.get_data()
		img_affine = img.affine
		print(sf)
		print('The img shape', img_np.shape[2])
		for i in range(img_np.shape[2]):
			slice_img_np = img_np[:,:,i]
			nft_img = nib.Nifti1Image(slice_img_np, img_affine)
			nib.save(nft_img, slice_dir_path + 'FLAIR_' + str(i) + '.nii.gz')

			if os.path.basename(sf) == '0':
				slice_img = nib.load(slice_dir_path + 'FLAIR_' + str(i) + '.nii.gz').get_data() / 5
				print('DID I GET HERE?')
				print('Writing to', str(i) + '.jpg')

def generateGTSlicesFolder(data_folder = '../Data/MS2017a/scans/'):
	scan_folders = glob.glob(data_folder + '*')

	for sf in scan_folders:
		slice_dir_path = os.path.join(sf, 'gt_slices/')
		if not os.path.exists(slice_dir_path):
			print('Creating directory at:' , slice_dir_path)
			os.makedirs(slice_dir_path)

		img = nib.load(os.path.join(sf, 'wmh.nii.gz'))
		img_np = img.get_data()
		img_affine = img.affine
		print(sf)
		print('The img shape', img_np.shape[2])
		for i in range(img_np.shape[2]):
			slice_img_np = img_np[:,:,i]
			nft_img = nib.Nifti1Image(slice_img_np, img_affine)
			nib.save(nft_img, slice_dir_path + 'wmh_' + str(i) + '.nii.gz')

			if os.path.basename(sf) == '0':
				slice_img = nib.load(slice_dir_path + 'wmh_' + str(i) + '.nii.gz').get_data() * 256
				#cv2.imwrite('temp/' + str(i) + '.jpg', slice_img)

def splitTrainVal_Slices(train_fraction, data_folder = '../Data/MS2017a/scans/'):
	scan_folders = glob.glob(data_folder + '/*/slices/*')
	num_scans = len(scan_folders)

	indices = np.random.permutation(num_scans)
	train_indices = indices[0:int(num_scans*train_fraction)]
	val_indices = indices[int(num_scans*train_fraction):]

	train_scan_folders = [scan_folders[i] for i in train_indices]
	val_scan_folders = [scan_folders[i] for i in val_indices]

	return train_scan_folders, val_scan_folders

def generateTrainValFile_Slices(train_fraction, main_folder = '../Data/MS2017a/'):
	train_folders, val_folders = splitTrainVal_Slices(0.8)

	train_folder_names = [train_folders[i].split(main_folder)[1] for i in range(len(train_folders))]
	val_folder_names = [val_folders[i].split(main_folder)[1] for i in range(len(val_folders))]

	f_train = open(main_folder + 'train_slices.txt', 'w+')
	f_val = open(main_folder + 'val_slices.txt', 'w+')

	for fn in train_folder_names:
		f_train.write(fn + '\n')

	for fn in val_folder_names:
		f_val.write(fn + '\n')

	f_train.close()
	f_val.close()

#Use this to save the images quickly (for testing purposes)
def quickSave(img, wmh, gif, n):
    nft_img = nib.Nifti1Image(img.squeeze(), np.eye(4))
    nib.save(nft_img, n + '_img.nii.gz')
    nft_img = nib.Nifti1Image(wmh.squeeze(), np.eye(4))
    nib.save(nft_img, n + '_wmh.nii.gz')
    if gif is not None:
        nft_img = nib.Nifti1Image(gif.squeeze(), np.eye(4))
        nib.save(nft_img, n + '_gif.nii.gz')

#------------------------------------------------------------------------------
#END OF SLICES PREPROCESSING
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#3D PREPROCESSING
#------------------------------------------------------------------------------
#go through every 3D object from training set and every patch of size NxNxN
#save resulting 3D volumes in one of the two folders based on what the center pixel of the image is

def extractCenterPixelPatches(N = 33, main_folder = '../Data/MS2017b/', postfix = ''):
	if N % 2 != 1:
		print('N must be odd')
		sys.exit()

	img_path = 'pre/FLAIR' + postfix + '.nii.gz'
	segm_path = 'wmh' + postfix + '.nii.gz'

	folders = ['lesion', 'other']
	patch_folder_path = os.path.join(main_folder, 'centerPixelPatches' + postfix + '_' + str(N))
	if not os.path.exists(patch_folder_path):
		for f in folders:
			os.makedirs(os.path.join(main_folder, patch_folder_path, f))

	scan_folders = glob.glob(main_folder + 'scans/*')

	counter = 0

	f_lesion_txt = open(os.path.join(patch_folder_path, 'lesion', 'center_locs.txt'), 'w+')
	f_other_txt = open(os.path.join(patch_folder_path, 'other', 'center_locs.txt'), 'w+')

	#This is only for training data
	img_list = read_file(main_folder + 'train' + postfix + '.txt', pretext = main_folder)
	print('Gathering training images from ' + main_folder + 'train' + postfix + '.txt')
	#remove pre/FLAIR_s.nii.gz from path. Only want up to folder name
	scan_folders = [img_list[i][:-len(img_path)] for i in range(len(img_list))]

	num_lesion = 0
	num_other = 0
	num_background = 0

	for sf in scan_folders:
		folder_num = sf.split('/')[-2]
		#read the FLAIR img
		img = nib.load(os.path.join(sf, img_path))
		img_affine = img.affine
		img_np = img.get_data()

		#read the wmh img
		wmh = nib.load(os.path.join(sf, segm_path))
		wmh_affine = wmh.affine
		wmh_np = wmh.get_data()

		#reshape to size1 x size2 -> size1 x size2 x 1
		img_np = img_np.reshape([img_np.shape[0], img_np.shape[1], img_np.shape[2], 1])
		wmh_np = wmh_np.reshape([wmh_np.shape[0], wmh_np.shape[1], wmh_np.shape[2], 1])

		#loop through every size
		for x in range(img_np.shape[0] - N + 1):
			for y in range(img_np.shape[1] - N + 1):
				for z in range(img_np.shape[2] - N + 1):
					wmh_patch = wmh_np[x:x+N, y:y+N, z:z+N]
					M = (N + 1) / 2
					center_pixel = wmh_patch[M,M,M]

					#folder_num | x | y | z
					location_name = str(folder_num) + '|' + str(x) + '|' + str(y) + '|' + str(z)
					if center_pixel == 1:
						num_lesion += 1
						f_lesion_txt.write(location_name + '\n')
				 	elif center_pixel == 2:
				 		num_other += 1
				 		f_other_txt.write(location_name + '\n')
		counter += 1
		print(str(counter) + ' / ' + str(len(scan_folders)))
	f_lesion_txt.close()
	f_other_txt.close()
	print('Num background: ', num_background)
	print('Num lesion', num_lesion)
	print('Num other', num_other)
	print('Done!')
	
		#TEMPORARY
		#if sf.split('/')
#during training we will sample uniformly between the two folders (uniformly select folder and uniformly select training sample)

def extractPatchBatch(batch_size, patch_size, img_list, 
						onlyLesions = False, center_pixel = False, 
						main_folder_path = '../../Data/MS2017b/', 
						postfix = '', with_priv = False):
	img_b = np.zeros([batch_size, 1, patch_size, patch_size, patch_size])
	label_b = np.zeros([batch_size, 1, patch_size, patch_size, patch_size])	

	gif_b = None
	if with_priv:
		gif_b = np.zeros([batch_size, 1, patch_size, patch_size, patch_size])	
		
	for i in range(batch_size):
		if center_pixel:
			center_pixel_folder_path = main_folder_path + 'centerPixelPatches' + postfix + '_' + str(patch_size)
			locs_lesion = open(os.path.join(center_pixel_folder_path, 'lesion', 'center_locs.txt')).readlines()
			locs_other =  open(os.path.join(center_pixel_folder_path, 'other', 'center_locs.txt')).readlines()
			img_patch, gt_patch, gif_patch = getCenterPixelPatch(patch_size, img_list, locs_lesion, locs_other, 
															onlyLesions, main_folder_path, postfix, with_priv)
		else:
			img_patch, gt_patch, gif_patch = getRandomPatch(patch_size, img_list, onlyLesions, main_folder_path, postfix, with_priv)
		
		img_b[i, :,:,:,:] = img_patch
		label_b[i, :,:,:,:] = gt_patch

		if with_priv:
			gif_b[i, :,:,:,:] = gif_patch
	return img_b, label_b, gif_b	

##################################################################################################################################
##################################################################################################################################
##################################Patch functions####################################################################
def getRandomPatch(patch_size, img_list, onlyLesions, main_folder_path, postfix, with_priv = False):
	img_str = img_list[randint(0, len(img_list)- 1)].rstrip()
	gt_str = img_str.replace('slices', 'gt_slices').replace('FLAIR', 'wmh').replace('/pre','')
	img_np = numpyFromScan(os.path.join(main_folder_path, img_str))
	gt_np = numpyFromScan(os.path.join(main_folder_path, gt_str), makebin = onlyLesions)

	img_np = img_np.transpose(3,0,1,2)
	gt_np = gt_np.transpose(3,0,1,2)
	img_dims = img_np.shape

	x = randint(0, img_dims[1]-patch_size-1)
	y = randint(0, img_dims[2]-patch_size-1)
	z = randint(0, img_dims[3]-patch_size-1)

	img_np_patch = img_np[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]
	gt_np_patch = gt_np[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]

	if with_priv:
		gif_str = img_str.replace('scans', 'gifs').replace('FLAIR','parcellation').replace('/pre','')
		gif_np = numpyFromScan(os.path.join(main_folder_path, gif_str))
		gif_np = gif_np.transpose(3,0,1,2)
		gif_np_patch = gif_np[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]
		return img_np_patch, gt_np_patch, gif_np_patch
	#draw 3 numbers between patch_size 
	return img_np_patch, gt_np_patch, None

#XXX not implemented for onlyLesions = True
def getCenterPixelPatch(patch_size, img_list, locs_lesion, locs_other, 
							onlyLesions, main_folder_path, postfix, with_priv = False):
	b = random.uniform(0.5, 3.5)
	#segm class = 1
	if b < 1.5:
		loc_str = locs_lesion[randint(0, len(locs_lesion) - 1)].rstrip()
	#segm class = 2
	elif b > 1.5 and b < 2.5 and (not onlyLesions):
		loc_str = locs_other[randint(0,len(locs_other) - 1)].rstrip()
	#segm class = 3
	else:
		loc_str = getBackgroundLoc(patch_size, img_list, onlyLesions, main_folder_path)

	#extract patch given folder number, location of top left edge and patch size
	#---------------------------------------------------------------------------
	folder_num_str, x, y, z = parseLocStr(loc_str)
	img_type_path = 'pre/FLAIR' + postfix + '.nii.gz'
	gt_type_path = 'wmh' + postfix + '.nii.gz'

	#read the file
	img_np = numpyFromScan(os.path.join(main_folder_path, 'scans', folder_num_str, img_type_path))
	gt_np = numpyFromScan(os.path.join(main_folder_path, 'scans', folder_num_str, gt_type_path), makebin = onlyLesions)

    #extract the patch
	patch_img_np = img_np[x:x+patch_size, y:y+patch_size, z:z+patch_size, :]
	patch_gt_np = gt_np[x:x+patch_size, y:y+patch_size, z:z+patch_size, :]
	
	#reshape to 1 x dim1 x dim2 x dim3
	patch_img_np = patch_img_np.transpose((3,0,1,2))
	patch_gt_np = patch_gt_np.transpose((3,0,1,2))

	if with_priv:
		gif_type_path = 'parcellation' + postfix + '.nii.gz'
		gif_np = numpyFromScan(os.path.join(main_folder_path, 'gifs', folder_num_str, gif_type_path))
		patch_gif_np = gif_np[x:x+patch_size, y:y+patch_size, z:z+patch_size, :]
		patch_gif_np = patch_gif_np.transpose((3,0,1,2))
		
		return patch_img_np, patch_gt_np, patch_gif_np
	return patch_img_np, patch_gt_np, None

def getBackgroundLoc(patch_size, img_list, onlyLesions, main_folder_path):
	num_generated = 0
	found_background = False

	#choose a random 3D image
	img_str = img_list[randint(0, len(img_list)- 1)].rstrip()
	curr_wmh_str = img_str.replace('slices', 'gt_slices').replace('FLAIR', 'wmh').replace('/pre','')
	wmh_np = numpyFromScan(os.path.join(main_folder_path, curr_wmh_str), makebin = onlyLesions)
	img_dims = wmh_np.shape
	folder_num = curr_wmh_str.split('/')[1]

	#print('THE FOLDER NUM', folder_num)
	while not found_background:
		x = randint(0, img_dims[0]-patch_size-1)
		y = randint(0, img_dims[1]-patch_size-1)
		z = randint(0, img_dims[2]-patch_size-1)

		#Load and check center pixel
		if wmh_np[x + ((patch_size - 1)/2), y + ((patch_size-1)/2), z + ((patch_size-1)/2)] == 0:
			found_background = True
			loc_str = str(folder_num) + '|' + str(x) + '|' + str(y) + '|' + str(z)
			return loc_str
		num_generated += 1
	#print('Num generated until a background batch was found: ', num_generated)
	return loc_str

def parseLocStr(loc_str):
    s = loc_str.split('|')
    return s[0], int(s[1]), int(s[2]), int(s[3])

##################################################################################################################################
##################################################################################################################################

#[518803341, 1496491, 217508]
def classCount(main_folder_path = '../Data/MS2017b/', img_type_path = 'pre/FLAIR_s.nii.gz', gt_type_path = 'wmh_s.nii.gz'):
		scan_folders = glob.glob(main_folder_path + 'scans/*')
		nums = [0, 0 ,0]
		for sf in scan_folders:
			wmh_np = numpyFromScan(os.path.join(sf, gt_type_path))
			unique, counts = np.unique(wmh_np, return_counts= True)
			d = dict(zip(unique, counts))
			for i in range(3):
				try: 
					nums[i] += d[i]
				except KeyError:
					pass
		print nums

def extractImgBatch(batch_size, img_list, img_size, onlyLesions = False, main_folder_path = '../Data/MS2017b/', with_priv = False):
	img_b = np.zeros([batch_size, 1, img_size[0], img_size[1], img_size[2]])
	label_b = np.zeros([batch_size, 1, img_size[0], img_size[1], img_size[2]])
	if with_priv:
		gif_b = np.zeros([batch_size, 1, img_size[0], img_size[1], img_size[2]])

	for i in range(batch_size):
		img_str = img_list[randint(0, len(img_list)-1)]
		img_np = numpyFromScan(os.path.join(main_folder_path, img_str))
		img_np = img_np.transpose((3,0,1,2))
		img_b[i, :,:,:,:] = img_np

		wmh_str = img_str.replace('slices', 'gt_slices').replace('FLAIR', 'wmh').replace('/pre','')
		gt_np = numpyFromScan(os.path.join(main_folder_path, wmh_str))
		gt_np = gt_np.transpose((3,0,1,2))
		label_b[i, :,:,:,:] = gt_np

		if with_priv:
			gif_str = img_str.replace('scans','gifs').replace('FLAIR', 'parcellation').replace('/pre','')
			gif_np = numpyFromScan(os.path.join(main_folder_path, gif_str))
			gif_np = gt_np.transpose((3,0,1,2))
			gif_b[i, :,:,:,:] = gif_np
	if with_priv:
		return img_b, label_b, gif_b
	return img_b, label_b, None

#------------------------------------------------------------------------------
#END OF 3D PREPROCESSING
#------------------------------------------------------------------------------

#3D
#generateTrainValFile(0.8, img_path = '/pre/FLAIR_s.nii.gz', main_folder = '../Data/MS2017b/')
#generateTrainValFile(0.8, img_path = '/pre/FLAIR_256x256x256orig.nii.gz', main_folder = '../Data/MS2017b/', postfix='_256x256x256orig')
#generateTrainValFile(0.8, main_folder = '../../Data/MS2017b/', postfix='_200x200x100orig')
#extractCenterPixelPatches(N = 81, main_folder = '../../Data/MS2017b/', postfix = '_200x200x100orig')
#extractCenterPixelPatches(N = 71, main_folder = '../../Data/MS2017b/', postfix = '_200x200x100orig')
#printDimensions(img_path = 'pre/FLAIR_s128.nii.gz', segm_path = 'wmh_s128.nii.gz', data_folder = '../Data/MS2017b/')
#printDimensions(img_path = 'pre/FLAIR.nii.gz', segm_path = 'wmh.nii.gz', data_folder = '../../Data/MS2017b/')
#printDimensions(img_path = 'pre/FLAIR256x256x256orig.nii.gz', segm_path = 'wmh.nii.gz', data_folder = '../../Data/MS2017b/')
#extractCenterPixelPatches()
#extractCenterPixelPatches(N = 91)
#extractCenterPixelPatches(N = 101, main_folder = '../../Data/MS2017b/', postfix = '_256x256x256orig')
#generateTrainValFile(0.8, img_path = '/pre/FLAIR_s128.nii.gz', main_folder = '../Data/MS2017b/', postfix='128x128x64')
#classCount()

#extractMeanDataStats()