import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet_2D
from collections import OrderedDict
import os
from os import walk
import torch.nn as nn
from docopt import docopt
sys.path.append('../')
import PP


docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot prefix [default: MRIslices_]
    --resultPath=<str>          Text file name to save results to  [default: ../evaluation_info/snapshots_results/results1.txt]
    --snapshotPath=<str>        Snapshot path [default: ../models/snapshots/]
    --mainFolderPath=<str>      Main folder path [default: ../../Data/MS2017a/]
    --LISTpath=<str>            Input image number list file [default: ../../Data/MS2017a/val_slices.txt]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 2]
    --gpu0=<int>                GPU number [default: 0]
    --useGPU=<int>              Use GPU [default: 0]
"""

args = docopt(docstr, version='v0.1')
print(args)

#intersection over union
def get_iou(pred,gt):
    if pred.shape != gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(args['--NoLabels'])-1  # labels from 0,1, ... 20(for VOC)  
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        #get list of tuple cordinates where j is the value
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))

        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)

        #if j exists in ground truth
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    #take average IOU over clasees
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 

    return Aiou


gpu0 = int(args['--gpu0'])
useGPU = int(args['--useGPU'])
snapPrefix = args['--snapPrefix'] 
test_list = args['--LISTpath']
results_txt_path = args['--resultPath']
snapshots_path = args['--snapshotPath']
main_folder_path = args['--mainFolderPath']

results_file = open(results_txt_path, 'w+')

if int(args['--NoLabels']) == 2:
    onlyLesions = True
else:
    onlyLesions = False

model = deeplab_resnet_2D.Res_Deeplab(int(args['--NoLabels']))
model.eval()

if useGPU:
    model.cuda(gpu0)

img_list = open(test_list).readlines()

for iter in range(1,21):
    if useGPU:
        #loading on GPU when model was saved on GPU
        saved_state_dict = torch.load(os.path.join(snapshots_path,snapPrefix+str(iter*1000)+'.pth'))
    else:
        #loading on CPU when model was saved on GPU
        saved_state_dict = torch.load(os.path.join(snapshots_path,snapPrefix+str(iter*1000)+'.pth'), map_location=lambda storage, loc: storage)

    model.load_state_dict(saved_state_dict)
    pytorch_list = [];
    counter = 0
    for img_str in img_list:
        try:
            #print(img_str[:-1])
            img = np.zeros((513, 513,1));
            img_temp = PP.numpyFromScan(os.path.join(main_folder_path, img_str[:-1]))

            img_original = img_temp
            img[:img_temp.shape[0],:img_temp.shape[1], :] = img_temp
            
            gt_str = img_str.replace('slices', 'gt_slices').replace('FLAIR', 'wmh')
            gt = PP.numpyFromScan(os.path.join(main_folder_path, gt_str[:-1]), makebin = onlyLesions)
        except IOError:
            continue

        if useGPU:
            output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
        else:
            output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True))

        interp = nn.UpsamplingBilinear2d(size=(513, 513))

        #interpolate the 4rd output which is the max taken over all 3 scales
        output = interp(output[3]).cpu().data[0].numpy()
        #take output pixels up to img_temp shape
        #shape of this is num_labels x dim1 x dim2
        output = output[:,:img_temp.shape[0],:img_temp.shape[1]]
        
        #transpose to be output : dim1 x dim2 x num_labels
        output = output.transpose(1,2,0)
        #take the argmax over the num_labels to get final predictions
        output = np.argmax(output, axis = 2)
        
        #dim1 x dim2 x 1 - > dim1 x dim2
        img_original = img_original.squeeze()
        #dim1 x dim2 x 1 -> dim1 x dim2
        gt = gt.squeeze()
        #dim1 x dim2 -> dim1 x dim2 (no reason just for consistency)
        output = output.squeeze()

        if args['--visualize']:
            plt.gray()
            plt.subplot(3, 1, 1)
            plt.imshow(img_original)
            plt.subplot(3, 1, 2)
            plt.imshow(gt)
            plt.subplot(3, 1, 3)
            plt.imshow(output)
            plt.show()

        iou_pytorch = get_iou(output,gt)    
        pytorch_list.append(iou_pytorch)
        counter += 1
        print "Model Iter | {:5d} | Progress: | {:4d}/{:4d}     \r".format(iter * 1000, counter, len(img_list)),
        sys.stdout.flush()
    results_file.write('Iterations: {:5d} | IOU : {:1.4f} \n'.format(iter*1000, np.sum(np.asarray(pytorch_list))/len(pytorch_list)))
print('Done!')
results_file.close()