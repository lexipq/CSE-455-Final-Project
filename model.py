# code for the model
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

basic_transformer = transforms.Compose([transforms.Resize((150,150)), transforms.ToTensor()])

class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        # images are 150x150x3
        self.conv_layer = nn.Conv2d(3, 8, 3)
        # after convolutional layer 148x148x8
        self.pool = nn.MaxPool2d(2)
        # afer one maxpool layer 74x74x8
        self.fc1 = nn.Linear(74*74*8, 250)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(batch_size, 74*74*8)
        x = self.fc1(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()
        in_channels = 3
        # size of the input image
        out_size = 150
        self.layers = nn.Sequential()
        for val in arr:
            if isinstance(val, int):
                # each convolutional layer we decrease the output size by 2
                # for kernel size 3
                out_size -= 2
                self.layers.append(nn.Conv2d(in_channels, val, 3))
                self.layers.append(nn.ReLU())
                # input channels are now the output channels
                in_channels = val
            elif isinstance(val, str) and val == "pool":
                # we floor divide the output size by 2 when we add a maxpool layer
                out_size //= 2
                self.layers.append(nn.MaxPool2d(2))
        # fully connected layer at the end
        self.fcl = nn.Linear(in_channels * out_size * out_size, 250)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layers(x)
        x = x.reshape(batch_size, self.fcl.in_features)
        x = self.fcl(x)
        return x

class ResNet_18(nn.Module):
    def __init__(self, arr=[]):
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.conv3 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 2))
        self.conv4 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256, 2))
        self.conv5 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 250)

    def forward(self, x):
        batch_size = x.size(0)
        # 7x7 kernel first layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # all residual block layers
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # average pool layers after residual layers
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # save identity value
        identity = x

        # first layer & activation
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        # second layer
        out = self.conv2(out)
        out = self.bn2(out)

        # paper starts downsampling from conv3_1
        if self.downsample is not None:
            identity = self.downsample(x)

        # add identity, then activate
        out += identity
        out = self.relu(out)
        return out
