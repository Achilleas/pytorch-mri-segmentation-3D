import numpy as np
import sys
import os
#implementation of normalization functions
	#In-subject Variance normalization, 
	#Global variance normalization
	#Histogram normalization (in use as default)

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#Histogram normalization functions

############################################################
#TRAINING
def applyHistNormalize(img, postfix, main_folder_path):
	file_path = os.path.join(main_folder_path, 'extra_data/', 'hist.txt')
	pc, s, m_p, mean_m = readHistInfo(file_path)

	return getTransform(img, pc, s, m_p, mean_m)

#returns landmark scores of the image
def getLandmarks(img, pc = (1,99), m_p=tuple(range(10, 100, 10))):
	img = img[img != 0] #and np.isfinite(img)
	threshold = np.mean(img)
	img = img[img > threshold]

	p = tuple(np.percentile(img, pc))

	m = tuple(np.percentile(img, m_p))

	return p, m

#extract linear map from p to s and map m's
#p is [p_1, p_2]
#s is [s_1, s_2]
#m is the landmark value
def mapLandmarksVec(p, s, m):
	p_1, p_2 = p[0], p[1]
	s_1, s_2 = s[0], s[1]

	new_val = np.zeros_like(p_1)
	same_inds = (p_1 == p_2)
	if np.sum(same_inds):
		print('Fix this')
		sys.exit()
		#Change with this if I encounter bug
		#new_val[same_inds] = s_1[same_inds].reshape(-1)
		#new_val[np.inverse(same_inds)] = (((m - p_1) * ((s_2 - s_1) / (p_2 - p_1))) + s_1).reshape(-1)

	#sys.exit()
	#new_val = ((m - p_1) * ((s_2 - s_1) / (p_2 - p_1))) + s_1

	return ((m-p_1) / (p_2-p_1) * (s_2 - s_1)) + s_1

def mapLandmarks(p, s, m):
	p_1, p_2 = p[0], p[1]
	s_1, s_2 = s[0], s[1]

	if p_1 == p_2:
		return s_1
	m_slope = (m-p_1) / (p_2-p_1)

	return (m_slope * (s_2 - s_1)) + s_1

##################################################################
def getTransform(img, pc, s, m_p, mean_m):
	z = np.copy(img)
	p, m = getLandmarks(img, pc, m_p)

	#using img, p, m, s, mean_m get the normalized image
	p_1, p_2 = p[0], p[1]
	s_1, s_2 = s[0], s[1]

	#histogram values at locations (pc + landmarks)
	m = [p_1] + list(m) + [p_2]
	#map scale corresponding to these values
	mean_m = [s_1] + list(mean_m) + [s_2]
	new_img = np.zeros_like(img, dtype=np.int64)
	hist_indices = np.zeros_like(img, dtype=np.int64)

	hist_indices = np.copy(new_img)

	for m_ in m:
		hist_indices += (img > m_).astype(int)

	hist_indices = np.clip(hist_indices, 1, len(m) - 1, out=hist_indices)

	indexer_m = lambda v: m[v]
	indexer_mm = lambda v: mean_m[v]
	f_m = np.vectorize(indexer_m)
	f_mm = np.vectorize(indexer_mm)
	
	new_p_1 = f_m(hist_indices - 1)
	new_p_2 = f_m(hist_indices)
	new_s_1 = f_mm(hist_indices - 1)
	new_s_2 = f_mm(hist_indices)
	
	new_img = mapLandmarksVec([new_p_1, new_p_2], [new_s_1, new_s_2], img)
	
	new_img = np.clip(new_img, s_1-1, s_2+1, out=new_img)
	
	return new_img

##################################################################

def iterOut(i):
    return ' '.join(str(x) for x in i) + '\n'

#READ AND WRITE
def writeHistInfo(filepath, pc, s, m_p, mean_m):
	with open(filepath, 'w+') as f:
		#parameters
		f.write(iterOut(pc))
		f.write(iterOut(s))
		f.write(iterOut(m_p))
		#result
		f.write(iterOut(mean_m))

def readHistInfo(filepath):
	lines = [line.rstrip() for line in open(filepath)]
	info = []
	info.append(tuple(float(x) for x in lines[0].split()))
	info.append(tuple(float(x) for x in lines[1].split()))
	info.append(tuple(float(x) for x in lines[2].split()))
	info.append(tuple(int(x) for x in lines[3].split()))
	
	#return     pc,		s, 		m_p, 	mean_m
	return info[0], info[1], info[2], info[3]
################################################################

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

#insubjectvar, globalvar supported so far
def applyNormalize(img, postfix, 
					norm_method = 'hist', 
					main_folder_path = '../../Data/MS2017b/'):
	if norm_method == 'insubjectvar':
		return applyInSubjectNormalize(img)
	elif norm_method == 'globalvar':
		return applyGlobalNormalize(img, postfix, main_folder_path)
	elif norm_method == 'hist':
		return applyHistNormalize(img, postfix, main_folder_path)
	else:
		print('Apply normalize doesn\'t support other functions currently')
		sys.exit()

def applyGlobalNormalize(img, postfix, main_folder_path = '../../Data/MS2017b/'):
	#subtract by dataset mean and divide by pixel standard deviation
	means = np.load(main_folder_path + 'extra_data/mean' + postfix + '.npy')
	stds = np.load(main_folder_path + 'extra_data/std' + postfix + '.npy')
	img = (img - means) / (stds + 0.000001)
	return img

def applyInSubjectNormalize(img):
	m = np.mean(img[img != 0])
	s = np.std(img[img != 0])
	img = (img - m) / s
	return img