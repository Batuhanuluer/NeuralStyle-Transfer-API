import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

class ResidualBlock(nn.Module):
    """
    Residual Block to maintain image details while deepening the network.
    Using InstanceNorm2d instead of BatchNorm2d for better style transfer results.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True) # FIXED NAME
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True) # FIXED NAME

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual # Skip connection
        return out


class Normalization(torch.nn.Module):
    """
    Normalize an input image using ImageNet statistics.
    VGG16 expects this specific normalization.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view(-1, 1, 1) transforms [3] -> [3, 1, 1] for broadcasting
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Move mean and std to the same device as the image
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # Load pre-trained VGG16 trained on ImageNet
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # We slice the model to get outputs from specific layers:
        # relu1_2, relu2_2, relu3_3, relu4_3
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Freeze the model (we don't train VGG)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        # Return a namedtuple for easy access to layers
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # 1. Initial Conv Layers (Downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True), # FIXED NAME
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True), # FIXED NAME
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True), # FIXED NAME
            nn.ReLU(inplace=True)
        )

        # 2. Residual Layers (Processing features)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # 3. Upsampling Layers (Expanding back to original size)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True), # FIXED NAME
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True), # FIXED NAME
            nn.ReLU(inplace=True)
        )
        
        # 4. Final Layer
        self.final = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)