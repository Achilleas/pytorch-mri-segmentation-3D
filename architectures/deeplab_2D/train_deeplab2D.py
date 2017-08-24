import torch
import torch.nn as nn
import numpy as np
import deeplab_resnet_2D
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import *
import random
from docopt import docopt
import sys

sys.path.append('../')
sys.path.append('../utils/')
import PP
#import matplotlib.pyplot as plt

docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --LISTpath=<str>            Input image number list file [default: ../../Data/MS2017a/train_slices.txt]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 2]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --useGPU=<int>              Use GPU or not [default: 0]
    --snapshotPath=<str>        Path to save snapshots [default: ../models/snapshots/]
    --snapshotPrefix=<str>      Snapshot model prefix [default: 2Ddeeplab_]
    --modelPath=<str>           Path of model to continue training on [default: none]
"""

args = docopt(docstr, version='v0.1')
print(args)

gpu0 = int(args['--gpu0'])
useGPU = int(args['--useGPU'])
snapshot_path = args['--snapshotPath']
snapshot_prefix = args['--snapshotPrefix']
model_path = args['--modelPath']
max_iter = int(args['--maxIter']) 
batch_size = 1
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

if int(args['--NoLabels']) == 2:
    onlyLesions = True
else:
    onlyLesions = False

if useGPU:
    cudnn.enabled = True
else:
    cudnn.enabled = False


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 """
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return j

def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

#returns a generator, returning the next chunk based on batch size
def chunker(seq, size):
 return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)
   
#Example chunk = ['scans/132/slices/FLAIR_28.nii.gz']
def get_data_from_chunk_v2(chunk):

    main_folder_path = '../../Data/MS2017a/'
    scans_folder_path = main_folder_path + 'scans/'

    img_type_path = 'pre/FLAIR.nii.gz'
    gt_type_path = 'wmh.nii.gz'

    scale = random.uniform(0.5, 1.3)
    dim = int(scale*321)

    images = np.zeros((dim,dim, 1,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    for i, piece in enumerate(chunk):
        print(os.path.join(main_folder_path, piece))
        img_temp = PP.numpyFromScan(os.path.join(main_folder_path, piece))
        flip_p = random.uniform(0, 1)

        img_temp = cv2.resize(img_temp,(321,321)).astype(float)
        img_temp = img_temp.reshape([321, 321, 1])

        img_temp = scale_im(img_temp,scale)
        img_temp = flip(img_temp,flip_p)
        images[:,:,0,i] = img_temp

        piece_gt = piece.replace('slices', 'gt_slices').replace('FLAIR', 'wmh')
        gt_temp = PP.numpyFromScan(os.path.join(main_folder_path, piece_gt), makebin = onlyLesions)
        gt_temp = cv2.resize(gt_temp,(321,321) , interpolation = cv2.INTER_NEAREST)
        gt_temp = gt_temp.reshape([321,321, 1])
        gt_temp = scale_gt(gt_temp,scale)
        gt_temp = flip(gt_temp,flip_p)

        gt[:,:,0,i] = gt_temp
        a = outS(321*scale)

    labels = [resize_label_batch(gt,i) for i in [a,a,a,a]]

    #from dim1 x dim2 x 1 x batch -> batch x 1 x dim1 x dim2
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    return images, labels

def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)

    label = torch.from_numpy(label).long()
    if useGPU:
        label = Variable(label).cuda(gpu0)
        if onlyLesions:
            criterion = nn.NLLLoss2d(weight = torch.cuda.FloatTensor([1, 100000]))
        else:
            criterion = nn.NLLLoss2d(weight = torch.cuda.FloatTensor([1, 100000, 100000]))
    else:
        label = Variable(label)

        if onlyLesions:
            criterion = nn.NLLLoss2d(weight = torch.FloatTensor([1, 100000]))
        else:
            criterion = nn.NLLLoss2d(weight = torch.FloatTensor([1, 100000, 100000]))

    m = nn.LogSoftmax()
    out = m(out)

    return criterion(out,label)


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def modelInit():
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    model = deeplab_resnet_2D.Res_Deeplab(int(args['--NoLabels']))

    if model_path != 'none':
        if useGPU:
            #loading on GPU when model was saved on GPU
            saved_state_dict = torch.load(model_path)
        else:
            #loading on CPU when model was saved on GPU
            saved_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state_dict)

    model.float()
    model.eval() # use_global_stats = True

    return model


def trainModel(model):
    img_list = read_file(args['--LISTpath'])

    data_list = []
    for i in range(50):  # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    if useGPU:
        model.cuda(gpu0)

    #criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

    optimizer.zero_grad()
    data_gen = chunker(data_list, batch_size)

    for iter in range(max_iter+1):
        chunk = data_gen.next()
        images, label = get_data_from_chunk_v2(chunk)

        if useGPU:
            images = Variable(images).cuda(gpu0)
        else:
            images = Variable(images)

        out = model(images)

        loss = loss_calc(out[0], label[0],gpu0)
        iter_size = int(args['--iterSize']) 
        for i in range(len(out)-1):
            loss = loss + loss_calc(out[i+1],label[i+1],gpu0)
        loss = loss/iter_size 
        loss.backward()

        if iter % 1 == 0:
            print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

        if iter % iter_size  == 0:
            optimizer.step()
            lr_ = lr_poly(base_lr,iter,max_iter,0.9)
            print '(poly lr policy) learning rate',lr_
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
            optimizer.zero_grad()

        if iter % 1000 == 0 and iter!=0:
            print 'taking snapshot ...'
            torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_prefix + str(iter) + '.pth'))

if __name__ == "__main__":
    model = modelInit()
    trainModel(model)    