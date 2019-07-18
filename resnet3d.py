import torch
import torch.nn as nn
import numpy as np

def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_3d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_3d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_3d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_3d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
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
 
def resnet3D50(**kwargs):
    return ResNet3d(Bottleneck3d, [3, 4, 6, 3], **kwargs)

def resnet3D101(**kwargs):
    return ResNet3d(Bottleneck3d, [3, 4, 23, 3], **kwargs)

def resnet3D152(**kwargs):
    return ResNet3d(Bottleneck3d, [3, 8, 36, 3], **kwargs)

def resnext3D50_32x4d( **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet3d(Bottleneck3d, [3, 4, 6, 3], **kwargs)

def resnext3D101_32x8d( **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet3d(Bottleneck3d, [3, 4, 23, 3], **kwargs)

class ResNet3DRegressor(nn.Module):
    def __init__(self):
        super(ResNet3DRegressor, self).__init__()
        self.resnet = resnet3D50(num_classes=512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc2(x)

        return x
    
class PipelinedResNet3d(ResNet3d):
    def __init__(self, block, layers, devices, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(PipelinedResNet3d, self).__init__(block, layers, num_classes, zero_init_residual,
        groups, width_per_group, replace_stride_with_dilation)
        assert( len(devices) == 2 and torch.cuda.is_available() )
        devices = ['cuda:{}'.format(device) for device in devices]
        self.dev1, self.dev2 = devices
        self.conv1    =  self.conv1.to(self.dev1)
        self.bn1      =  self.bn1.to(self.dev1)
        self.relu     =  self.relu.to(self.dev1)
        self.maxpool  =  self.maxpool.to(self.dev1)
        self.layer1   =  self.layer1.to(self.dev1)
        self.layer2   =  self.layer2.to(self.dev2)
        self.layer3   =  self.layer3.to(self.dev2)
        self.layer4   =  self.layer4.to(self.dev2)
        self.avgpool  =  self.avgpool.to(self.dev2)
        self.fc       =  self.fc.to(self.dev2)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x = x.to(self.dev2)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        x = x.to(self.dev1)
        return x



def pipelined_resnet3D50(devices, **kwargs):
    return PipelinedResNet3d(Bottleneck3d, [3, 4, 6, 3], devices, **kwargs)


class PipelinedResNet3dRegressor(nn.Module):
    def __init__(self, devices):
        super(PipelinedResNet3dRegressor, self).__init__()
        self.pipelinedresnet = pipelined_resnet3D50(devices, num_classes=512)
        self.fc2 = nn.Linear(512, 1).to(devices[0])
    
    def forward(self, x):
        x = self.pipelinedresnet(x)
        x = self.fc2(x)

        return x