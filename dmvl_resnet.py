import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['dmvl_resnet32', 'dmvl_resnet110']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, number_net=4, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.number_net = number_net

        self.dilation = 1
        self.inplanes = 16
        self.number_net = number_net
        self.num_classes = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group


        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        fix_planes = self.inplanes

        for i in range(self.number_net):
            self.inplanes = fix_planes
            setattr(self, 'layer2_share' + str(i), self._make_layer(block, 32, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[1]))
            setattr(self, 'layer3_share' + str(i), self._make_layer(block, 64, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[2]))
            setattr(self, 'classifier_share' + str(i), nn.Linear(64 * block.expansion, self.num_classes))

        for i in range(self.number_net):
            self.inplanes = fix_planes
            setattr(self, 'layer2_specific' + str(i), self._make_layer(block, 32, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[1]))
            setattr(self, 'layer3_specific' + str(i), self._make_layer(block, 64, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[2]))
            setattr(self, 'classifier_specific' + str(i), nn.Linear(64 * block.expansion, self.num_classes))

        self.discriminator = nn.Sequential(
            nn.Linear(64 * block.expansion, self.number_net),
        )

        self.information_fusion = nn.Sequential(
            nn.Linear((1+self.number_net)*64, (1+self.number_net)*64),
            nn.BatchNorm1d((1+self.number_net)*64)
        )

        self.classifier_fusion = nn.Sequential(
            nn.Linear((1+self.number_net)*64, (1+self.number_net)*64),
            nn.BatchNorm1d((1+self.number_net)*64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear((1+self.number_net)*64, self.num_classes),
            nn.BatchNorm1d(self.num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
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


    def _forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        input = x
        embedding = []
        logits_share = []
        x_share_view_codes =[]

        logits_view_share = []
        logits_view_specific = []

        x_common = 0.

        for i in range(self.number_net):

            x_share = getattr(self, 'layer2_share' + str(i))(input)
            x_share = getattr(self, 'layer3_share' + str(i))(x_share)
            x_share = self.avgpool(x_share)
            embedding.append(x_share)
            x_share = x_share.view(x_share.size(0), -1)
            x_share_view_code = torch.ones(x_share.shape[0],1)*i 
            x_share_view_codes.append(x_share_view_code)
            logits_share.append(self.discriminator(x_share)) 
            logit_view_share = getattr(self, 'classifier_share' + str(i))(x_share)  
            logits_view_share.append(logit_view_share)
            x_common = torch.add(x_common,x_share) 

        x_common = x_common/self.number_net
        x_fusion = x_common
        regularization = 0.
        x_specific_features = []
        for i in range(self.number_net):

            x_specific = getattr(self, 'layer2_specific' + str(i))(input)
            x_specific = getattr(self, 'layer3_specific' + str(i))(x_specific)
            x_specific = self.avgpool(x_specific)
            x_specific = x_specific.view(x_specific.size(0), -1)
            logit_view_specific = getattr(self, 'classifier_specific' + str(i))(x_specific)
            logits_view_specific.append(logit_view_specific)
            x_fusion = torch.cat((x_fusion, x_specific), 1)
            x_specific_features.append(x_specific)
            regularization += view_specific_common_regularization(x_specific, x_common)


        feature_fusion = self.information_fusion(x_fusion)
        logits = getattr(self, 'classifier_fusion')(feature_fusion)
        x_specific_features = torch.cat(x_specific_features, 1)
        regularization_specific = view_specific_regularization(x_specific_features, self.number_net)
        return logits, embedding, regularization, logits_share, x_share_view_codes, logits_view_share, x_common, x_fusion, feature_fusion, regularization_specific, logits_view_specific

    forward = _forward





def view_specific_common_regularization(view_feature, comm_feature):

    item = view_feature * comm_feature
    item = item.sum(1)
    item = item ** 2
    loss = item.sum()
    loss /= comm_feature.shape[0]
    return loss


def view_specific_regularization(view_specific_fatures, num_branches):
    view_specific = torch.chunk(view_specific_fatures, num_branches,dim=1)
    loss = 0.
    for i in range(len(view_specific)):
        for j in range(len(view_specific)):
            if i < j:
                item = torch.mm(view_specific[i], view_specific[j].T)
                item = item ** 2
                item = item.sum()
                loss += item
    loss /= (view_specific[0].shape[0] * view_specific[0].shape[0])
    return loss

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def dmvl_resnet32(**kwargs):
    return _resnet(BasicBlock, [5, 5, 5], **kwargs)

def dmvl_resnet110(**kwargs):
    return _resnet(Bottleneck, [12, 12, 12], **kwargs)

if __name__ == '__main__':
    net = dmvl_resnet32(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits, embedding = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
