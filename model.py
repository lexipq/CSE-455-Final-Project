# code for the model
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

basic_transformer = transforms.Compose([transforms.ToTensor()])

class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        # images are 256x256x3
        self.conv_layer = nn.Conv2d(3, 8, 3)
        # after convolutional layer 254x254x8
        self.pool = nn.MaxPool2d(2)
        # afer one maxpool layer 127x127x8
        self.fc1 = nn.Linear(127*127*8, 5)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(batch_size, self.fc1.in_features)
        x = self.fc1(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()
        in_channels = 3
        # size of the input image
        out_size = 256
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
        self.fcl = nn.Linear(in_channels * out_size * out_size, 5)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layers(x)
        x = x.reshape(batch_size, self.fcl.in_features)
        x = self.fcl(x)
        return x
