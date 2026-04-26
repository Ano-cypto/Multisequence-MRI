# discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.norm4 = nn.InstanceNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        

    def forward(self, x):
        features = []

        x = self.leaky_relu1(self.conv1(x))
        features.append(x)

        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        features.append(x)

        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        features.append(x)

        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        features.append(x)

        out = self.conv5(x)

        return out, features

