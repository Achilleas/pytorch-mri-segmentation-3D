import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=0, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, padding=0,batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, padding=0,batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, padding=0, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, padding=0,batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, padding=0,batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, padding=0,batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False,padding=0, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9_up = self.dc9(e7)
        d9 = torch.cat([d9_up, self.center_crop(syn2, d9_up.size()[2:5])], 1)
        del d9_up, e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        dc6_up = self.dc6(d7)
        d6 = torch.cat([dc6_up, self.center_crop(syn1, dc6_up.size()[2:5])], 1)
        del dc6_up, d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        dc3_up = self.dc3(d4)
        d3 = torch.cat([dc3_up, self.center_crop(syn0, dc3_up.size()[2:5])],1)
        del dc3_up, d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2
        d0 = self.dc0(d1)

        #interpolate to original output dimensions
        s0 = x.size()[2]
        s1 = x.size()[3]
        s2 = x.size()[4]
        self.interp = nn.Upsample(size = (s0, s1, s2), mode='trilinear')
        d0 = self.interp(d0)
        return d0

    def center_crop(self, layer, target_sizes):
        batch_size, n_channels, dim1, dim2, dim3 = layer.size()
        dim1_c = (dim1 - target_sizes[0]) // 2
        dim2_c = (dim2 - target_sizes[1]) // 2
        dim3_c = (dim3 - target_sizes[2]) // 2
        return layer[:, :, dim1_c:dim1_c+target_sizes[0], dim2_c:dim2_c+target_sizes[1], dim3_c:dim3_c+target_sizes[2]]