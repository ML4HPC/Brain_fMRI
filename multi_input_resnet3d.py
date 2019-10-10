import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet3d import Bottleneck3d, conv1x1_3d, conv3x3_3d

class ResNet3dPre(nn.Module):
    """ 
        only upto layer 1 
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3dPre, self).__init__()
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        return x

def resnet3d_pre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer):
    return ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)


class ResNet3dPrePool(nn.Module):
    """ 
        only upto layer 1 
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3dPrePool, self).__init__()
        self.resnet3dpre = resnet3d_pre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.avgpool = nn.AdaptiveAvgPool3d((48, 48, 48))


    def forward(self, x):
        x = self.resnet3dpre(x)
        x = self.avgpool(x)

        return x

class ResNet3dPost(nn.Module):
    """ 
        the next 3 layers 
    """

    def __init__(self, block, layers, num_classes=1000, inplanes=718, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3dPost, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class OneStructInputMultiOutputResNet3d(nn.Module):
    """
        require four devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=768, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(OneStructInputMultiOutputResNet3d, self).__init__()
        assert( len(devices) >= 3 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[0])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        # Age
        self.fc1    =  nn.Linear(1000, 1).to(self.devs[2])
        # Gender
        self.fc2    =  nn.Linear(1000, 1).to(self.devs[2])
        # Race
        self.fc3    =  nn.Linear(1000, 5).to(self.devs[2])
        # Education
        self.fc4    =  nn.Linear(1000, 21).to(self.devs[2])
        # Income
        self.fc5    =  nn.Linear(1000, 10).to(self.devs[2])
        # Married
        self.fc6    =  nn.Linear(1000, 6).to(self.devs[2])
        # ABCD site
        self.fc7    =  nn.Linear(1000, 22).to(self.devs[2])
        # Volume
        self.fc8    =  nn.Linear(1000, 1).to(self.devs[2])
        # Height
        self.fc9    =  nn.Linear(1000, 1).to(self.devs[2])
        # Weight
        self.fc10    =  nn.Linear(1000, 1).to(self.devs[2])
        # BMI
        self.fc11    =  nn.Linear(1000, 1).to(self.devs[2])
        # Fluid intelligence
        self.fc12    =  nn.Linear(1000, 1).to(self.devs[2])
        # Crystallized intelligence
        self.fc13    =  nn.Linear(1000, 1).to(self.devs[2])
        # Pattern score
        self.fc14    =  nn.Linear(1000, 1).to(self.devs[2])
        # Picture score
        self.fc15    =  nn.Linear(1000, 1).to(self.devs[2])
        # List score
        self.fc16    =  nn.Linear(1000, 1).to(self.devs[2])
        # Flanker score
        self.fc17    =  nn.Linear(1000, 1).to(self.devs[2])
        # Pic/Vocab score
        self.fc18    =  nn.Linear(1000, 1).to(self.devs[2])
        # Cards sort score
        self.fc19    =  nn.Linear(1000, 1).to(self.devs[2])
        # Total score
        self.fc20    =  nn.Linear(1000, 1).to(self.devs[2])
        # Reading score
        self.fc21    =  nn.Linear(1000, 1).to(self.devs[2])


    def forward(self, x):
        x = self.head(x)
        x = x.to(self.devs[1])

        x = self.tail(x)
        x = x.to(self.devs[2])

        x_final = []
        x_final.append(self.fc1(x).to(self.devs[0]))
        x_final.append(self.fc2(x).to(self.devs[0]))
        x_final.append(self.fc3(x).to(self.devs[0]))
        x_final.append(self.fc4(x).to(self.devs[0]))
        x_final.append(self.fc5(x).to(self.devs[0]))
        x_final.append(self.fc6(x).to(self.devs[0]))
        x_final.append(self.fc7(x).to(self.devs[0]))
        x_final.append(self.fc8(x).to(self.devs[0]))
        x_final.append(self.fc9(x).to(self.devs[0]))
        x_final.append(self.fc10(x).to(self.devs[0]))
        x_final.append(self.fc11(x).to(self.devs[0]))
        x_final.append(self.fc12(x).to(self.devs[0]))
        x_final.append(self.fc13(x).to(self.devs[0]))
        x_final.append(self.fc14(x).to(self.devs[0]))
        x_final.append(self.fc15(x).to(self.devs[0]))
        x_final.append(self.fc16(x).to(self.devs[0]))
        x_final.append(self.fc17(x).to(self.devs[0]))
        x_final.append(self.fc18(x).to(self.devs[0]))
        x_final.append(self.fc19(x).to(self.devs[0]))
        x_final.append(self.fc20(x).to(self.devs[0]))
        x_final.append(self.fc21(x).to(self.devs[0]))

        return x_final

    def get_devices(self):
        return self.devs

class OneDTIInputMultiOutputResNet3d(nn.Module):
    """
        require four devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=768, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(OneDTIInputMultiOutputResNet3d, self).__init__()
        assert( len(devices) == 3 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head   =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[2])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        # Age
        self.fc1    =  nn.Linear(1000, 1).to(self.devs[2])
        # Gender
        self.fc2    =  nn.Linear(1000, 1).to(self.devs[2])
        # Race
        self.fc3    =  nn.Linear(1000, 5).to(self.devs[2])
        # Education
        self.fc4    =  nn.Linear(1000, 21).to(self.devs[2])
        # Income
        self.fc5    =  nn.Linear(1000, 10).to(self.devs[2])
        # Married
        self.fc6    =  nn.Linear(1000, 6).to(self.devs[2])
        # ABCD site
        self.fc7    =  nn.Linear(1000, 22).to(self.devs[2])
        # Volume
        self.fc8    =  nn.Linear(1000, 1).to(self.devs[2])
        # Height
        self.fc9    =  nn.Linear(1000, 1).to(self.devs[2])
        # Weight
        self.fc10    =  nn.Linear(1000, 1).to(self.devs[2])
        # BMI
        self.fc11    =  nn.Linear(1000, 1).to(self.devs[2])
        # Fluid intelligence
        self.fc12    =  nn.Linear(1000, 1).to(self.devs[2])
        # Crystallized intelligence
        self.fc13    =  nn.Linear(1000, 1).to(self.devs[2])
        # Pattern score
        self.fc14    =  nn.Linear(1000, 1).to(self.devs[2])
        # Picture score
        self.fc15    =  nn.Linear(1000, 1).to(self.devs[2])
        # List score
        self.fc16    =  nn.Linear(1000, 1).to(self.devs[2])
        # Flanker score
        self.fc17    =  nn.Linear(1000, 1).to(self.devs[2])
        # Pic/Vocab score
        self.fc18    =  nn.Linear(1000, 1).to(self.devs[2])
        # Cards sort score
        self.fc19    =  nn.Linear(1000, 1).to(self.devs[2])
        # Total score
        self.fc20    =  nn.Linear(1000, 1).to(self.devs[2])
        # Reading score
        self.fc21    =  nn.Linear(1000, 1).to(self.devs[2])


    def forward(self, x):
        x = self.head(x)
        x = x.to(self.devs[1])

        x = self.tail(x)
        x = x.to(self.devs[2])

        x_final = []
        x_final.append(self.fc1(x).to(self.devs[0]))
        x_final.append(self.fc2(x).to(self.devs[0]))
        x_final.append(self.fc3(x).to(self.devs[0]))
        x_final.append(self.fc4(x).to(self.devs[0]))
        x_final.append(self.fc5(x).to(self.devs[0]))
        x_final.append(self.fc6(x).to(self.devs[0]))
        x_final.append(self.fc7(x).to(self.devs[0]))
        x_final.append(self.fc8(x).to(self.devs[0]))
        x_final.append(self.fc9(x).to(self.devs[0]))
        x_final.append(self.fc10(x).to(self.devs[0]))
        x_final.append(self.fc11(x).to(self.devs[0]))
        x_final.append(self.fc12(x).to(self.devs[0]))
        x_final.append(self.fc13(x).to(self.devs[0]))
        x_final.append(self.fc14(x).to(self.devs[0]))
        x_final.append(self.fc15(x).to(self.devs[0]))
        x_final.append(self.fc16(x).to(self.devs[0]))
        x_final.append(self.fc17(x).to(self.devs[0]))
        x_final.append(self.fc18(x).to(self.devs[0]))
        x_final.append(self.fc19(x).to(self.devs[0]))
        x_final.append(self.fc20(x).to(self.devs[0]))
        x_final.append(self.fc21(x).to(self.devs[0]))

        return x_final

    def get_devices(self):
        return self.devs


class TwoInputResNet3d(nn.Module):
    """
        require three devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=768, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(TwoInputResNet3d, self).__init__()
        assert( len(devices) == 3 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head1  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[0])
        self.head2  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[2])

    def forward(self, x):
        x1 = self.head1(x[0])
        x2 = self.head2(x[1])
        x1 = x1.to(self.devs[2])
        x2 = x2.to(self.devs[2])
        x  = torch.cat((x1,x2),1)
        x  = self.tail(x)
        x  = x.to(self.devs[0])
        return x

class ThreeInputResNet3d(nn.Module):
    """
        require four devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=768, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ThreeInputResNet3d, self).__init__()
        assert( len(devices) == 4 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head1  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[0])
        self.head2  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        self.head3  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[2])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[3])
        self.fc     =  nn.Linear(1000, 1).to(self.devs[3])

    def forward(self, x):
        x1 = self.head1(x[0])
        x2 = self.head2(x[1])
        x3 = self.head3(x[2])
        x1 = x1.to(self.devs[3])
        x2 = x2.to(self.devs[3])
        x3 = x3.to(self.devs[3])

        x  = torch.cat((x1,x2,x3),1)
        x  = self.tail(x)
        x  = self.fc(x)
        x  = x.to(self.devs[0])
        return x

    def get_devices(self):
        return self.devs



class SixInputResNet3d(nn.Module):
    """
        require four devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=1536, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SixInputResNet3d, self).__init__()
        assert( len(devices) == 7 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head1  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[0])
        self.head2  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        self.head3  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[2])
        self.head4  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[3])
        self.head5  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[4])
        self.head6  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[5])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[6])
        self.fc     =  nn.Linear(1000, 1).to(self.devs[6])

    def forward(self, x):
        x1 = self.head1(x[0])
        x2 = self.head2(x[1])
        x3 = self.head3(x[2])
        x4 = self.head4(x[3])
        x5 = self.head5(x[4])
        x6 = self.head6(x[5])

        x1 = x1.to(self.devs[6])
        x2 = x2.to(self.devs[6])
        x3 = x3.to(self.devs[6])
        x4 = x4.to(self.devs[6])
        x5 = x5.to(self.devs[6])
        x6 = x6.to(self.devs[6])

        x  = torch.cat((x1,x2,x3,x4,x5,x6),1)
        x  = self.tail(x)
        x  = self.fc(x)
        x  = x.to(self.devs[0])
        return x

    def get_devices(self):
        return self.devs


class SixInputMultiOutputResNet3d(nn.Module):
    """
        require four devices
    """
    def __init__(self, block, layers, devices, num_classes=1000, post_inplanes=1536, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SixInputMultiOutputResNet3d, self).__init__()
        assert( len(devices) == 8 and torch.cuda.is_available() )
        self.devs   =  ['cuda:{}'.format(device) for device in devices]
        self.head1  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[0])
        self.head2  =  ResNet3dPrePool(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[1])
        self.head3  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[2])
        self.head4  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[3])
        self.head5  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[4])
        self.head6  =  ResNet3dPre(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[5])
        self.tail   =  ResNet3dPost(block, layers, num_classes, post_inplanes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer).to(self.devs[6])
        # Age
        self.fc1    =  nn.Linear(1000, 1).to(self.devs[7])
        # Gender
        self.fc2    =  nn.Linear(1000, 1).to(self.devs[7])
        # Race
        self.fc3    =  nn.Linear(1000, 5).to(self.devs[7])
        # Education
        self.fc4    =  nn.Linear(1000, 21).to(self.devs[7])
        # Income
        self.fc5    =  nn.Linear(1000, 10).to(self.devs[7])
        # Married
        self.fc6    =  nn.Linear(1000, 6).to(self.devs[7])
        # ABCD site
        self.fc7    =  nn.Linear(1000, 22).to(self.devs[7])
        # Volume
        self.fc8    =  nn.Linear(1000, 1).to(self.devs[7])
        # Height
        self.fc9    =  nn.Linear(1000, 1).to(self.devs[7])
        # Weight
        self.fc10    =  nn.Linear(1000, 1).to(self.devs[7])
        # BMI
        self.fc11    =  nn.Linear(1000, 1).to(self.devs[7])
        # Fluid intelligence
        self.fc12    =  nn.Linear(1000, 1).to(self.devs[7])
        # Crystallized intelligence
        self.fc13    =  nn.Linear(1000, 1).to(self.devs[7])
        # Pattern score
        self.fc14    =  nn.Linear(1000, 1).to(self.devs[7])
        # Picture score
        self.fc15    =  nn.Linear(1000, 1).to(self.devs[7])
        # List score
        self.fc16    =  nn.Linear(1000, 1).to(self.devs[7])
        # Flanker score
        self.fc17    =  nn.Linear(1000, 1).to(self.devs[7])
        # Pic/Vocab score
        self.fc18    =  nn.Linear(1000, 1).to(self.devs[7])
        # Cards sort score
        self.fc19    =  nn.Linear(1000, 1).to(self.devs[7])
        # Total score
        self.fc20    =  nn.Linear(1000, 1).to(self.devs[7])
        # Reading score
        self.fc21    =  nn.Linear(1000, 1).to(self.devs[7])


    def forward(self, x):
        x1 = self.head1(x[0])
        x2 = self.head2(x[1])
        x3 = self.head3(x[2])
        x4 = self.head4(x[3])
        x5 = self.head5(x[4])
        x6 = self.head6(x[5])
        
        x1 = x1.to(self.devs[6])
        x2 = x2.to(self.devs[6])
        x3 = x3.to(self.devs[6])
        x4 = x4.to(self.devs[6])
        x5 = x5.to(self.devs[6])
        x6 = x6.to(self.devs[6])

        x  = torch.cat((x1,x2,x3,x4,x5,x6),1)
        x  = self.tail(x).to(self.devs[7])

        x_final = []
        x_final.append(self.fc1(x).to(self.devs[0]))
        x_final.append(self.fc2(x).to(self.devs[0]))
        x_final.append(self.fc3(x).to(self.devs[0]))
        x_final.append(self.fc4(x).to(self.devs[0]))
        x_final.append(self.fc5(x).to(self.devs[0]))
        x_final.append(self.fc6(x).to(self.devs[0]))
        x_final.append(self.fc7(x).to(self.devs[0]))
        x_final.append(self.fc8(x).to(self.devs[0]))
        x_final.append(self.fc9(x).to(self.devs[0]))
        x_final.append(self.fc10(x).to(self.devs[0]))
        x_final.append(self.fc11(x).to(self.devs[0]))
        x_final.append(self.fc12(x).to(self.devs[0]))
        x_final.append(self.fc13(x).to(self.devs[0]))
        x_final.append(self.fc14(x).to(self.devs[0]))
        x_final.append(self.fc15(x).to(self.devs[0]))
        x_final.append(self.fc16(x).to(self.devs[0]))
        x_final.append(self.fc17(x).to(self.devs[0]))
        x_final.append(self.fc18(x).to(self.devs[0]))
        x_final.append(self.fc19(x).to(self.devs[0]))
        x_final.append(self.fc20(x).to(self.devs[0]))
        x_final.append(self.fc21(x).to(self.devs[0]))

        return x_final

    def get_devices(self):
        return self.devs

def one_struct_input_multi_output_resnet3D50(devices, **kwargs):
    return OneStructInputMultiOutputResNet3d(Bottleneck3d,[3, 4, 6, 3], devices, **kwargs)

def one_dti_input_multi_output_resnet3D50(devices, **kwargs):
    return OneDTIInputMultiOutputResNet3d(Bottleneck3d,[3, 4, 6, 3], devices, **kwargs)

def bi_input_resnet3D50(devices, **kwargs):
    return TwoInputResNet3d(Bottleneck3d,[3, 4, 6, 3], devices, **kwargs)

def tri_input_resnet3D50(devices, **kwargs):
    return ThreeInputResNet3d(Bottleneck3d,[3, 4, 6, 3], devices, **kwargs)

def hex_input_multi_output_resnet3D50(devices, **kwargs):
    return SixInputMultiOutputResNet3d(Bottleneck3d,[3, 4, 6, 3], devices, **kwargs)
