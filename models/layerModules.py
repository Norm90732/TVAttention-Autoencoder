import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpylayers

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding,transpose,act=True):
        super(ConvolutionalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.transpose = transpose
        if transpose == True:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride),]
        if act== True:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(EncodingBlock, self).__init__()
        #Encoder Block
        self.convBlock1 = ConvolutionalBlock(in_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.convBlock2 = ConvolutionalBlock(out_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        skip = x
        x = self.p1(x)
        return x, skip


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(BottleneckBlock, self).__init__()
        self.convBlock1 = ConvolutionalBlock(in_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)
        self.convBlock2 = ConvolutionalBlock(out_channels, out_channels,kernel_size, stride, padding,transpose=False,act=True)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(DecoderBlock, self).__init__()
        self.convBlockup1 = ConvolutionalBlock(in_channels, out_channels,kernel_size=kernel_size, padding=padding, stride=stride,transpose=True,act=True)
        self.convBlock2 = ConvolutionalBlock(out_channels*2,out_channels,kernel_size=kernel_size, padding=padding, stride=stride,transpose=False,act=True)
        self.convBlock3 = ConvolutionalBlock(out_channels, out_channels,kernel_size=kernel_size, padding=padding, stride=stride,transpose=False,act=True)

    def forward(self,x,skip):
        x = self.convBlockup1(x)
        x = torch.cat([x,skip],dim=1)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        return x







