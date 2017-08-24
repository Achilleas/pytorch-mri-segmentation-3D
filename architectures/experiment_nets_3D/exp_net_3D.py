import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import sys
import torch.nn.init as init
affine_par = True

class HighResNetBlock(nn.Module):

    def __init__(self, inplanes, outplanes, padding_=1, stride=1, dilation_ = 1):
        super(HighResNetBlock, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=3, stride=1, 
                                padding=padding_, bias=False, dilation = dilation_)
        self.conv2 = nn.Conv3d(outplanes, outplanes, kernel_size=3, stride=1, 
                                padding=padding_, bias=False, dilation = dilation_)
        #2 convolutions of same dilation. residual block
        self.bn1 = nn.BatchNorm3d(outplanes, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.bn2 = nn.BatchNorm3d(outplanes, affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.relu = nn.PReLU()
        self.diff_dims = (inplanes != outplanes)

        self.downsample = nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(outplanes, affine = affine_par)
        )
        for i in self.downsample._modules['1'].parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.diff_dims:
            residual = self.downsample(residual)

        out += residual
        return out

class TripleBlock(nn.Module):
    def __init__(self, inplanes, outplanes, padding_=1, stride=1, dilation_ = 1):
        super(TripleBlock, self).__init__()
        self.block0 = HighResNetBlock(inplanes=inplanes, outplanes=outplanes, padding_=padding_, dilation_=dilation_)
        self.block1 = HighResNetBlock(inplanes=outplanes, outplanes=outplanes, padding_=padding_, dilation_=dilation_)
        self.block2 = HighResNetBlock(inplanes=outplanes, outplanes=outplanes, padding_=padding_, dilation_=dilation_)
    
    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)

        return out

class downSampleBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(downSampleBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(outplanes, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class ASPP_Module(nn.Module):

    def __init__(self,dilation_series, padding_series, inplanes, midplanes, outplanes):
        super(ASPP_Module, self).__init__()
        self.conv3d_list = nn.ModuleList()
        self.bn3d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv3d_list.append(nn.Conv3d(inplanes, midplanes,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
            self.bn3d_list.append(nn.BatchNorm3d(midplanes, affine = affine_par))
        self.num_concats = len(self.conv3d_list) + 2
        #add global pooling, add batchnorm
        self.conv1x1_1 = nn.Conv3d(inplanes, midplanes, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv3d(inplanes, midplanes, kernel_size=1, stride=1)
        self.conv1x1_3 = nn.Conv3d(midplanes*self.num_concats, outplanes, kernel_size=1, stride=1)

        self.relu = nn.PReLU()

        self.bn1 = nn.BatchNorm3d(midplanes, affine = affine_par)
        self.bn2 = nn.BatchNorm3d(midplanes*self.num_concats, affine= affine_par)
        self.bn3 = nn.BatchNorm3d(midplanes, affine= affine_par)

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
        out = self.relu(out)

        #apply 1x1 convolution to get back to output number filters
        out =  self.conv1x1_3(out)
        #apply last batch norm
        out  = self.bn3(out)
        out = self.relu(out)
        return out

#replaced transpose convolutions with trilinear interpolation
class expNet(nn.Module):
    def __init__(self, NoLabels1, dilations, isPriv, NoLabels2 = 209, withASPP = True):
        super(expNet,self).__init__()
        d = dilations

        self.isPriv = isPriv
        self.NoLabels2 = NoLabels2
        self.withASPP = withASPP

        self.downsample0 = downSampleBlock(inplanes = 1, outplanes = 16)

        self.block1_1 = TripleBlock(inplanes=16, outplanes=16, padding_=d[0], dilation_=d[0])
        self.block1_2 = TripleBlock(inplanes=16, outplanes=32, padding_=d[1], dilation_=d[1])

        self.downsample1 = downSampleBlock(inplanes = 32, outplanes = 32)

        self.block2_1 = TripleBlock(inplanes=32, outplanes=64, padding_=d[2], dilation_=d[2])

        self.block2_2_b2 = TripleBlock(inplanes=64, outplanes=128, padding_=d[3], dilation_=d[3])
        self.upsample_b2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.classifier_b2 = nn.Conv3d(64 + 32, NoLabels1, kernel_size=1, stride=1, padding=0, bias=False)

        #branch 1 is priv branch
        if self.isPriv:
            self.block2_2_b1 = TripleBlock(inplanes=64, outplanes=128, padding_=d[3], dilation_=d[3])
            self.upsample_b1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
            self.classifier_b1 = nn.Conv3d(64 + 32, NoLabels2, kernel_size=1, stride=1, padding=0, bias=False)
            if self.withASPP:
                self.aspp_b2 = ASPP_Module([4,8,12],[4,8,12], inplanes = 256, midplanes = 128, outplanes=128)
            else:
                self.upsample_b2 = nn.ConvTranspose3d(256, 64, kernel_size=2, stride=2, bias=False)
        else:
            if self.withASPP:
                self.aspp_b2 = ASPP_Module([4,8,12],[4,8,12], inplanes = 128, midplanes = 128, outplanes=128)

    def forward(self, x):
        s0 = x.size()[2]
        s1 = x.size()[3]
        s2 = x.size()[4]
        self.interp = nn.Upsample(size = (s0, s1, s2), mode='trilinear')

        out = self.downsample0(x)
        paddings = (int(out.size()[2] % 2), int(out.size()[3] % 2), int(out.size()[4] % 2))

        out = self.block1_1(out)
        out_s = self.block1_2(out)

        out = self.downsample1(out_s)
        out = self.block2_1(out)

        if not self.isPriv:
            out = self.block2_2_b2(out)
            if self.withASPP:
                out = self.aspp_b2(out)
            out = self.upsample_b2(out)[:,:,paddings[0]:, paddings[1]:, paddings[2]:]
            out = torch.cat((out, out_s), 1)
            out = self.classifier_b2(out)
            out = self.interp(out)
            return out
        else:
            out1 = self.block2_2_b1(out)
            out2 = self.block2_2_b2(out)
            del out

            out2 = torch.cat((out2, out1), 1)
            if self.withASPP:
                out2 = self.aspp_b2(out2)
            out2 = self.upsample_b2(out2)[:,:,paddings[0]:, paddings[1]:, paddings[2]:]
            out2 = torch.cat((out2, out_s), 1)
            out2 = self.classifier_b2(out2)

            out1 = self.upsample_b1(out1)[:,:,paddings[0]:, paddings[1]:, paddings[2]:]
            out1 = torch.cat((out1, out_s), 1)
            out1 = self.classifier_b1(out1)

            out1 = self.interp(out1)
            out2 = self.interp(out2)
            return out1, out2

#dilations is contains 6 elements
def getExpNet(NoLabels1, dilations, isPriv, NoLabels2 = 209, withASPP = True):
    model = expNet(NoLabels1, dilations, isPriv, NoLabels2 = NoLabels2, withASPP = withASPP)

    for m in model.modules():
        if isinstance(m,nn.Conv3d):
            init.kaiming_uniform(m.weight)
        elif isinstance(m, nn.Sequential):
            for m_1 in m.modules():
                if isinstance(m_1, nn.Conv3d):
                    init.kaiming_uniform(m_1.weight)
    return model
