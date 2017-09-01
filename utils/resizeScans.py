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
import subprocess
from docopt import docopt
import PP
import normalizations as NORM
import augmentations as AUGM
#this utility resizes images to a specific dimension given by the user
#NOTE: this does not register the images to a template, only resizes them

#pass flair folder
#will find and resize the rest .. for wmh, ../gifs for gifs

docstr = """Write something here

Usage: 
    resizeScans.py [options]

Options:
    -h, --help                  Print this message
    --mainFolderPath=<str>      Main folder path [default: ../../Data/MS2017b/]
    --FLAIR_name=<str>          FLAIR prefix [default: FLAIR.nii.gz]
    --WMH_name=<str>            WMH prefix [default: wmh.nii.gz]
    --GIF_name=<str>            GIF prefix [default: parcellation.nii.gz]
    --CIRC_name=<str>           CIRC prefix [default: circumference.nii.gz]
    --gpu0=<int>                GPU number [default: 0]
    --useGPU=<int>              Use GPU [default: 0]
    --size=<str>                Shape to resize to [default: 200x200x100]
    --noGIFS                    Dont Resize GIFS
    --postfix=<str>             Postfix name to add [default: _200x200x100orig]
    --withNorm=<int>            Include normalization [default: 1]
"""


args = docopt(docstr, version='v0.1')
#resize256x256x256
noGIFS = args['--noGIFS']
main_folder_path = args['--mainFolderPath']

FLAIR_name = args['--FLAIR_name']
WMH_name = args['--WMH_name']
GIF_name = args['--GIF_name']
CIRC_name = args['--CIRC_name']
postfix = args['--postfix']
norm_type = ['hist', 'insubjectvar']
with_norm = int(args['--withNorm'])
glob_folders = glob.glob(main_folder_path + 'scans/*')

new_size = args['--size'].split('x')
new_size = np.array([int(i) for i in new_size])


def convertSize(from_path, to_path, new_size, interpolation = 'interpolate'):
	img = PP.numpyFromScan(from_path)
	shape = img.shape
	r1 = shape[0] / float(new_size[0])
	r2 = shape[1] / float(new_size[1])
	r3 = shape[2] / float(new_size[2])
	command = "mri_convert " + from_path + " " + to_path + " -ds " + str(r1) + " " + str(r2) + " " + str(r3) + " -rt " + interpolation
	print(command)
	#normalize
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	process.communicate()

def convertSize2(from_path, to_path, new_size, interpolation = 'interpolate'):
	if interpolation == 'interpolate':
		spline_order = [2]
	elif interpolation == 'nearest':
		spline_order = [0]

	img_np, affine = PP.numpyFromScan(from_path, get_affine = True)
	shape = img_np.shape
	new_affine = np.copy(affine)
	r1 = float(new_size[0]) / shape[0]
	r2 = float(new_size[1]) / shape[1] 
	r3 = float(new_size[2]) / shape[2] 
	new_affine[:,0] /= r1
	new_affine[:,1] /= r2
	new_affine[:,2] /= r3

	img_np = AUGM.applyScale([img_np], [r1,r2,r3], spline_order)[0].squeeze()

	PP.saveScan(img_np, new_affine, to_path)
	return new_affine

def normalizeScan(from_path, to_path, main_folder_path = main_folder_path):
	img_np, affine = PP.numpyFromScan(from_path, get_affine = True)
	img_np = NORM.applyNormalize(img_np.squeeze(), postfix, norm_method = norm_type[0], main_folder_path = main_folder_path)
	img_np = NORM.applyNormalize(img_np.squeeze(), postfix, norm_method = norm_type[1], main_folder_path = main_folder_path)
	PP.saveScan(img_np, affine, to_path)

if __name__ == "__main__":

	for scan_folder in glob_folders:
		folder_num = os.path.basename(scan_folder)

		flair_path = os.path.join(main_folder_path, 'scans', folder_num, 'pre', FLAIR_name)
		wmh_path = os.path.join(main_folder_path, 'scans', folder_num, WMH_name)
		gif_path = os.path.join(main_folder_path, 'gifs', folder_num, GIF_name)
		circ_path = os.path.join(main_folder_path, 'gifs', folder_num, CIRC_name)

		to_flair_path = flair_path.replace('.nii.gz', postfix + '.nii.gz')
		to_wmh_path = wmh_path.replace('.nii.gz', postfix + '.nii.gz')
		to_gif_path = gif_path.replace('.nii.gz', postfix + '.nii.gz')
		to_circ_path = circ_path.replace('.nii.gz', postfix + '.nii.gz')

		convertSize2(flair_path, to_flair_path, new_size)
		if with_norm:
			normalizeScan(to_flair_path, to_flair_path)

		convertSize2(wmh_path, to_wmh_path, new_size, 'nearest')
		if not noGIFS:
			convertSize2(gif_path, to_gif_path, new_size, 'nearest')