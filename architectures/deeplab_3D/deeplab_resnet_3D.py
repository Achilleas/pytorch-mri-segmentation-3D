import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation_
        #if dilation_ == 2:
        #    padding = 2
        #elif dilation_ == 4:
        #    padding = 4
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm3d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv3d_list = nn.ModuleList()
        self.bn3d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv3d_list.append(nn.Conv3d(2048, 256,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
            self.bn3d_list.append(nn.BatchNorm3d(256, affine = affine_par))
        self.num_concats = len(self.conv3d_list) + 2
        #add global pooling, add batchnorm
        self.conv1x1_1 = nn.Conv3d(2048, 256, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv3d(2048, 256, kernel_size=1, stride=1)
        self.conv1x1_3 = nn.Conv3d(256*self.num_concats, 256, kernel_size=1, stride=1)
        self.conv1x1_4 = nn.Conv3d(256, NoLabels, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm3d(256, affine = affine_par)
        self.bn2 = nn.BatchNorm3d(256*self.num_concats, affine= affine_par)
        self.bn3 = nn.BatchNorm3d(256, affine= affine_par)
        #global avg pool
        #input = 1x512xdim1xdim2xdim3
        #output = 1x512x1x1x1
        #XXX check

        for m in self.conv3d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.bn3d_list[0](self.conv3d_list[0](x))
        #concatenate multiple atrous rates
        for i in range(len(self.conv3d_list)-1):
            #XXX add batch norm?
            out = torch.cat([out, self.bn3d_list[i+1](self.conv3d_list[i+1](x))], 1)

        #concatenate global avg pooling (avg global pool -> 1x1 conv (256 filter) -> batchnorm -> interpolate -> concat)
        self.glob_avg_pool = nn.AvgPool3d(kernel_size=(x.size()[2],x.size()[3],x.size()[4]))
        self.iterp_orig = nn.Upsample(size = (out.size()[2], out.size()[3], out.size()[4]), mode= 'trilinear')

        out = torch.cat([out, self.iterp_orig(self.bn1(self.conv1x1_1(self.glob_avg_pool(x))))], 1)
        
        #concatenate 1x1 convolution
        out = torch.cat([out, self.conv1x1_2(x)], 1)

        #apply batch norm on concatenated output
        out = self.bn2(out)
        
        #apply 1x1 convolution to get back to 256 filters
        out =  self.conv1x1_3(out)

        #apply last batch norm
        out  = self.bn3(out)

        #apply 1x1 convolution to get last labels
        out =  self.conv1x1_4(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False) # / 2
        self.bn1 = nn.BatchNorm3d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # / 4
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=1) # / 8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1) # / 16

        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 8)
        self.layer4_3 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 16)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                for i in m.parameters():
                    i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion,affine = affine_par),
            )
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        #print('A - x', x.size())
        x = self.conv1(x)
        #print('B - conv1', x.size())
        x = self.bn1(x)
        #print('C - bn1', x.size())
        x = self.relu(x)
        #print('D - relu', x.size())
        x = self.maxpool(x)
        #print('E - maxpool', x.size())
        x = self.layer1(x)
        #print('F - layer1', x.size())
        x = self.layer2(x)
        #print('G - layer2', x.size())
        x = self.layer3(x)
        #print('H - layer3', x.size())
        x = self.layer4_0(x)
        #print('I - layer4_0', x.size())
        x = self.layer4_1(x)
        #print('J - layer4_1', x.size())
        x = self.layer4_2(x)
        #print('K - layer4_2', x.size())
        x = self.layer4_3(x)
        #print('L - layer4_3', x.size())
        x = self.layer5(x)
        #print('M - layer5 (classification)', x.size())
        return x
#
class MS_Deeplab(nn.Module):
    def __init__(self,block,NoLabels):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3], NoLabels)

    def forward(self,x):
        s0 = x.size()[2]
        s1 = x.size()[3]
        s2 = x.size()[4]
        #self.interp3 = nn.Upsample(size = ( outS(s0), outS(s1), outS(s2) ), mode= 'nearest')
        self.interp = nn.Upsample(size = (s0, s1, s2), mode='trilinear')
        out = self.interp(self.Scale(x))
        #out = []

        #add 1x1 convolution
        #add ASPP (6, 12, 18 dilations)
        #add avg max pooling with 1x1 conv
        #print('N - upsample', out.size())
        #out.append(self.Scale(x))
        return out

def Res_Deeplab(NoLabels=3):
    model = MS_Deeplab(Bottleneck,NoLabels)

    
    return model