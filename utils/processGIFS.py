import os
import sys
import glob

from shutil import copyfile

gif_folders_path = '../../../T1s/results/'
save_folder = '../../Data/MS2017b/gifs/'

gif_folders = glob.glob(gif_folders_path + '*')

print(gif_folders_path)

if not os.path.exists(save_folder):
	print('Creating directory at:' , save_folder)
	os.makedirs(save_folder)
for gif_folder in gif_folders:

	folder_num = os.path.basename(gif_folder).split('_')[1]

	if not os.path.exists(save_folder + folder_num):
		os.makedirs(save_folder + folder_num)

	parcellation_f = ''
	circumference_f = ''


	for f_name in glob.glob(gif_folder + '/*'):
		if 'Parcellation' in f_name:
			parcellation_f = f_name
		elif 'Segmentation' in f_name:
			circumference_f = f_name

	copyfile(os.path.join(gif_folder, parcellation_f), os.path.join(save_folder, folder_num, 'parcellation.nii.gz'))
	copyfile(os.path.join(gif_folder, circumference_f), os.path.join(save_folder, folder_num, 'circumference.nii.gz'))