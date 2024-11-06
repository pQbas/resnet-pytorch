import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1):
        super(ResNetBlock, self).__init__()


        self.w1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=1, 
                      bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.w2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.use_identity = (c_in != c_out)
        
        if self.use_identity:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, 
                                      bias=False)
        else:
            self.identity = nn.Identity()        

    def forward(self, x):
        
        F = nn.Sequential(
            self.w1,
            nn.ReLU(),
            self.w2)

        I = self.identity
        
        return F(x) + I(x)


class ResNet(nn.Module):
    def __init__(self, categories=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_1 = nn.Sequential(
            ResNetBlock(64, 64, kernel_size=3),
            nn.ReLU()
        )

        self.conv2_2 = nn.Sequential(
            ResNetBlock(64, 64, kernel_size=3),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            ResNetBlock(64, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.conv3_2 = nn.Sequential(
            ResNetBlock(128, 128, kernel_size=3),
            nn.ReLU()
        )

        self.conv4_1 = nn.Sequential(
            ResNetBlock(128, 256, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.conv4_2 = nn.Sequential(
            ResNetBlock(256, 256, kernel_size=3),
            nn.ReLU()
        )

        self.conv5_1 = nn.Sequential(
            ResNetBlock(256, 512, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.conv5_2 = nn.Sequential(
            ResNetBlock(512, 512, kernel_size=3),
            nn.ReLU()
        )

        self.decode = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, categories)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.decode(x)
        return x
