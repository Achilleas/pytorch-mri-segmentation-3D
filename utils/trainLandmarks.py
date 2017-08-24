import numpy as np
import glob
import sys
from docopt import docopt
import os
import PP
import normalizations as NORM
docstr = """Write something here 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --mainFolderPath=<str>      Main folder path [default: ../../Data/MS2017b/]
    --postfix=<str>             Postfix of flair images. i.e. to use FLAIR_s postfix is _s. This also determines the train file [default: ]
    --pcRange=<str>             PC range [default: 1-99]
"""

args = docopt(docstr, version='v0.1')
print(args)

main_folder_path=args['--mainFolderPath']
postfix = args['--postfix']
pc_range = args['--pcRange'].split('-')
save_folder = os.path.join(main_folder_path, 'extra_data/')
save_path = os.path.join(save_folder, 'hist' + postfix + '.txt')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

pc = (int(pc_range[0]), int(pc_range[1]))
m_p = tuple(range(10, 100, 10))
s = (34.681492767333985, 1638.193154296875)
#s = (34, 1638)

def trainLandmarks(main_folder_path = main_folder_path, postfix = postfix):
	scan_folders = glob.glob(main_folder_path + 'scans/*')
	FLAIR_path = '/pre/FLAIR' + postfix + '.nii.gz'
	m_arr = np.zeros([len(scan_folders), len(m_p)])

	for i, sf in enumerate(scan_folders):
		print "Landmark training: {:4d}/{:4d}\r".format(i, len(scan_folders)),
		sys.stdout.flush()

		img_str = sf + FLAIR_path
		img_np = PP.numpyFromScan(img_str)

		p, m = NORM.getLandmarks(img_np)
		mapped_m = np.array([int(NORM.mapLandmarks(p, s, x)) for x in m], dtype=np.int64)
		m_arr[i, :] = mapped_m

	mean_m = np.mean(m_arr, axis = 0, dtype=np.int64)

	NORM.writeHistInfo(save_path, pc, s, m_p, mean_m)




 #dwi.standardize.write_std_cfg(cfgpath, pc, landmarks, scale, mapped_scores,
 #                                 thresholding)
def getScale(main_folder_path = main_folder_path, postfix = postfix):
	scan_folders = glob.glob(main_folder_path + 'scans/*')

	FLAIR_path = '/pre/FLAIR' + postfix + '.nii.gz'
	min_p = None
	max_p = None
	for i, sf in enumerate(scan_folders):
		print "Scale obtaining: {:4d}/{:4d} 		\r".format(i, len(scan_folders)),
		sys.stdout.flush()

		img_str = sf + FLAIR_path
		img_np = PP.numpyFromScan(img_str)
		p, m = NORM.getLandmarks(img_np)
		if min_p is None:
			min_p = p[0]
			max_p = p[1]
		if min_p > p[0]:
			min_p = p[0]
		if max_p < p[1]:
			max_p = p[1]
	return (min_p, max_p)

'''

def get_stats(pc, scale, landmarks, img, thresholding):
    """Gather info from single image."""
    p, scores = dwi.standardize.landmark_scores(img, pc, landmarks,
                                                thresholding)
    p1, p2 = p
    s1, s2 = scale
    mapped_scores = [dwi.standardize.map_onto_scale(p1, p2, s1, s2, x) for x in
                     scores]
    mapped_scores = [int(x) for x in mapped_scores]
    return dict(p=p, scores=scores, mapped_scores=mapped_scores)


def train(pc, scale, landmarks, inpaths, cfgpath, thresholding, verbose):
    """Training phase."""
    data = []
    for inpath in inpaths:
        img, _ = dwi.files.read_pmap(inpath)
        if img.shape[-1] != 1:
            raise Exception('Incorrect shape: {}'.format(inpath))
        d = get_stats(pc, scale, landmarks, img, thresholding)
        if verbose:
            # print(img.shape, dwi.util.fivenum(img), inpath)
            # print(d['p'], d['scores'], inpath)
            print(img.shape, d['mapped_scores'], inpath)
        data.append(d)
    mapped_scores = np.array([x['mapped_scores'] for x in data], dtype=np.int)
    mapped_scores = np.mean(mapped_scores, axis=0, dtype=mapped_scores.dtype)
    mapped_scores = list(mapped_scores)
    if verbose:
        print(mapped_scores)
    dwi.standardize.write_std_cfg(cfgpath, pc, landmarks, scale, mapped_scores,
                                  thresholding)
'''



if __name__ == "__main__":
	#obtain the maximum and minimum intensities based on IOI (varies with different pc)
	print('Calculating scale...')
	s = getScale()
	print('Training landmarks...')
	trainLandmarks()
	print('Done!')
