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
