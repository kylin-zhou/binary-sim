import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channel, out_channel, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):

    """
    Basic building block for ResNet.

    Args：
        - in_channel: Number of input channel
        - out_channel: Number of output channel
        - stride: Number of stride
        - downsample: "None" for identity downsample, otherwise for a real downsample

    """

    expansion = (
        1  # Record whether the number of convolution kernels in each layer has changed
    )

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # Record the output of the last residual block

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if (
            self.downsample is not None
        ):  # Determine if need to downsample for dimension matching
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Implementation of ResNet architecture.

    Arg：
        - block: "BasicBlock" for ResNet-18/34, "Bottleneck" for ResNet-50/101/152
        - layers: The number of each residual layer, for example, [3,4,6,3] for the ResNet-34/50
        - num_classes: The number of labels in the data set
        - grayscale: "True" for single-channel images, "False" for 3-channel images

    """

    def __init__(self, block, layers, num_classes=64, grayscale=False):
        self.in_channel = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()

        #  conv1 + maxpooling
        self.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #  conv2,3,4,5
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #  avgpooling + fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #  Initialization of the convolutional layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    out_channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    """return a ResNet object"""
    return ResNet(BasicBlock, [2, 1, 1, 1], grayscale=True)
