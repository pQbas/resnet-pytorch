import torch
import torch.nn as nn
from collections import OrderedDict

class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1):
        super(ResNetBlock, self).__init__()

        self.F = nn.Sequential(
            # 1st convolutional block
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            # Activation function
            nn.ReLU(),
            # 2nd convolutional block
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.I = nn.Identity() if c_in == c_out else \
                 nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        self.A = nn.ReLU()

    def forward(self, x):
        F = self.F
        I = self.I
        A = self.A
        return A(F(x) + I(x))


class ResNet(nn.Module):
    def __init__(self, categories=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        backbone_layers = [
            ('conv2_1' , ResNetBlock(64,  64,  kernel_size=3)),
            ('conv2_2' , ResNetBlock(64,  64,  kernel_size=3)),
            ('conv3_1' , ResNetBlock(64,  128, kernel_size=3, stride=2)),
            ('conv3_2' , ResNetBlock(128, 128, kernel_size=3)),
            ('conv4_1' , ResNetBlock(128, 256, kernel_size=3, stride=2)),
            ('conv4_2' , ResNetBlock(256, 256, kernel_size=3)),
            ('conv5_1' , ResNetBlock(256, 512, kernel_size=3, stride=2)),
            ('conv5_2' , ResNetBlock(512, 512, kernel_size=3))
        ]
        self.backbone = nn.Sequential(OrderedDict(backbone_layers))
        
        self.decode = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, categories)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.backbone(x)
        x = self.decode(x)
        return x
