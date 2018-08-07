## TODO: define the convolutional neural network architecture
# make the input smaller

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, fileName=None):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input size 1* 100* 100
        
        self.conv1 = nn.Conv2d(1, 32, 4) 
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop1= nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.drop3 = nn.Dropout2d(p=0.3)
        self.drop4 = nn.Dropout2d(p=0.4)
        self.drop5 = nn.Dropout2d(p=0.5)
        self.drop6 = nn.Dropout2d(p=0.6)
        
        # (100 - 3) = 97 / 2 = 48 - 2 = 46 / 2 = 23 -1 = 22 /2 = 11/2 = 5
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.fc1 = nn.Linear(5*5*256, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.out = nn.Linear(1000, 136)
        if fileName != None:
            if torch.cuda.device_count() == 0:
                self.load_state_dict(torch.load(fileName, map_location='cpu'))
            else:
                self.load_state_dict(torch.load(fileName))
                
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        conv1 = self.drop1(self.pool(F.elu(self.conv1(x))))
        conv2 = self.drop2(self.pool(F.elu(self.conv2(conv1))))
        conv3 = self.drop3(self.pool(F.elu(self.conv3(conv2))))
        conv4 = self.drop4(self.pool(F.elu(self.conv4(conv3))))
        
        # flatten the data
        flatten = conv4.view(conv4.size(0), -1)
        fc1 = self.drop5(F.elu(self.fc1(flatten)))
        fc2 = self.drop6(F.relu(self.fc2(fc1)))
        output = self.out(fc2)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return output

