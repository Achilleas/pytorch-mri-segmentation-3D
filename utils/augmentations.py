import numpy as np
import nibabel as nib
import scipy.ndimage
import warnings
import PP
import sys

#---------------------------------------------
#Functions for image augmentations on 3D input
#---------------------------------------------

#img_b, label_b is (batch_num) x 1 x dim1 x dim2 x dim3
#takes in a list of 3D images (1st one is input, 2nd one needs to be label)
def augmentPatchLossy(imgs, rotation=[5,5,5], scale_min=0.9, scale_max=1.1, flip_lvl = 0):
	new_imgs = []

	rot_x = np.random.uniform(-rotation[0], rotation[0]) * np.pi / 180.0
	rot_y = np.random.uniform(-rotation[1], rotation[1]) * np.pi / 180.0
	rot_z = np.random.uniform(-rotation[2], rotation[2]) * np.pi / 180.0

	zoom_val = np.random.uniform(scale_min, scale_max)
	for i in range(len(imgs)):
		l = convertBatchToList(imgs[i])
		if i == 0:
			spline_orders = [3] * len(l)
		else:
			spline_orders = [0] * len(l)
		scaled = applyScale(l, zoom_val, spline_orders)
		rotated = applyRotation(scaled, [rot_x, rot_y, rot_z], spline_orders)
		new_imgs.append(convertListToBatch(rotated))
	return imgs

def convertBatchToList(img):
	l = []
	b, c, d1, d2, d3 = img.shape
	for i in range(img.shape[0]):
		l.append(img[i,:,:,:,:].reshape([1,c,d1,d2,d3]))
	return l

def convertListToBatch(img_list):
	b, c, d1, d2, d3 = img_list[0].shape
	a = np.zeros([len(img_list), c, d1,d2,d3])
	for i in range(len(img_list)):
		a[i,:,:,:,:] = img_list[i]
	return a

def augmentPatchLossLess(imgs):
	new_imgs = []

	p = np.random.rand(3) > 0.5
	locations = np.where(p == 1)[0] + 2

	for i in range(len(imgs)):
		l = convertBatchToList(imgs[i])
		if i == 0:
			spline_orders = [3] * len(l)
		else:
			spline_orders = [0] * len(l)
		flipped = applyFLIPS2(l, locations)

		rot_x = np.random.randint(4) * np.pi / 2.0 # (0,1,2,3)*90/180.0
		rot_y = np.random.randint(4) * np.pi / 2.0 # (0,1,2,3)*90/180.0
		rot_z = np.random.randint(4) * np.pi / 2.0 # (0,1,2,3)*90/180.0
		rotated = applyRotation(flipped, [rot_x, rot_y, rot_z], spline_orders)
		new_imgs.append(convertListToBatch(rotated))
	return new_imgs

def augmentBoth(imgs):
	imgs = augmentPatchLossy(imgs)
	imgs = augmentPatchLessLess(imgs)
	return imgs

def getRotationVal(rotation=[5,5,5]):
	rot_x = np.random.uniform(-rotation[0], rotation[0]) * np.pi / 180.0
	rot_y = np.random.uniform(-rotation[1], rotation[1]) * np.pi / 180.0
	rot_z = np.random.uniform(-rotation[2], rotation[2]) * np.pi / 180.0
	return rot_x, rot_y, rot_z

def getScalingVal(scale_min = 0.9, scale_max = 1.1):
	return np.random.uniform(scale_min, scale_max)

def applyFLIPS(images, flip_lvl = 0):
	if flip_lvl == 0:
		p = np.random.rand(2) > 0.5
	else:
		p = np.random.rand(3) > 0.5
	locations = np.where(p == 1)[0] + 2

	new_imgs = []
	for img in images:
		for i in locations:
			img = np.flip(img, axis=i)
		new_imgs.append(img)
	return new_imgs

def applyFLIPS2(images, locations):
	new_imgs = []
	for img in images:
		for i in locations:
			img = np.flip(img, axis=i)
		new_imgs.append(img)
	return new_imgs

def applyRotation(images, rot, spline_orders):
	transform_x = np.array([[1.0, 				0.0,			0.0],
                            [0.0, 				np.cos(rot[0]), -np.sin(rot[0])],
                            [0.0, 				np.sin(rot[0]), np.cos(rot[0])]])

	transform_y = np.array([[np.cos(rot[1]), 	0.0, 			np.sin(rot[1])],
                            [0.0, 				1.0, 			0.0],
                            [-np.sin(rot[1]), 	0.0, 			np.cos(rot[1])]])

	transform_z = np.array([[np.cos(rot[2]),	-np.sin(rot[2]), 	0.0],
                            [np.sin(rot[2]), 	np.cos(rot[2]), 	0.0],
                            [0.0, 				0, 					1]])
	transform = np.dot(transform_z, np.dot(transform_x, transform_y))

	new_imgs = []
	for i, img in enumerate(images):
		mid_index = 0.5 * np.asarray(img.squeeze().shape, dtype=np.int64)
		offset = mid_index - mid_index.dot(np.linalg.inv(transform))
		new_img = scipy.ndimage.affine_transform(
											input = img.squeeze(), 
											matrix = transform, 
											offset = offset, 
											order = spline_orders[i],
											mode = 'nearest')
		new_img = new_img[np.newaxis,np.newaxis,:]
		new_imgs.append(new_img)
	return new_imgs

def applyScale(images, zoom_val, spline_orders):
	new_imgs = []
	for i, img in enumerate(images):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			try:
				new_img = scipy.ndimage.zoom(img.squeeze(), zoom_val, order = spline_orders[i])
				new_img = new_img[np.newaxis,np.newaxis,:]
				new_imgs.append(new_img)
			except:
				pass
	return new_imgs