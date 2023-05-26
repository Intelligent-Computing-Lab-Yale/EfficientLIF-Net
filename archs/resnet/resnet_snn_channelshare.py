'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

tau_global = 1. / (1. - 0.25)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, num_chgroup, stride=1):
        super(BasicBlock, self).__init__()
        self.num_chgroup =num_chgroup

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,  bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1,  bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,  bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        act = self.bn1(self.conv1(x))

        N_list =[]
        for i in range(self.num_chgroup):
            out_temp = self.lif1(act[:,(act.size(1)//self.num_chgroup)*i:(act.size(1)//self.num_chgroup)*(i+1),...])
            N_list.append(out_temp)
        out = torch.cat(N_list, dim=1)


        out = self.bn2(self.conv2(out))
        act = out + self.shortcut(x)
        N_list = []

        for i in range(self.num_chgroup):
            out_temp = self.lif2(
                act[:, (act.size(1) // self.num_chgroup) * i:(act.size(1) // self.num_chgroup) * (i + 1), ...])
            N_list.append(out_temp)
        out = torch.cat(N_list, dim=1)

        return out




class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10, total_timestep=6, num_chgroup=2):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.total_timestep = total_timestep
        self.num_chgroup = num_chgroup

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)
        self.bn1 = nn.BatchNorm2d(128)

        self.layer1 = self._make_layer(block, 256, num_blocks[0], num_chgroup,stride=2)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], num_chgroup,stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], num_chgroup,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, num_chgroup,stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, num_chgroup,stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_img):

        output_list = []
        static_x = self.bn1(self.conv1(input_img))

        for t in range(self.total_timestep):

            N_list = []
            for i in range(self.num_chgroup):
                out_temp = self.lif_input(static_x[:, (static_x.size(1) // self.num_chgroup) * i:(static_x.size(
                    1) // self.num_chgroup) * (i + 1), ...])
                N_list.append(out_temp)
            out = torch.cat(N_list, dim=1)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)

            output_list.append(out)

        return output_list

def ResNet19_channelshare(args, num_classes, total_timestep,num_chgroup):
    return ResNet(args, BasicBlock, [3, 3, 2], num_classes, total_timestep, num_chgroup)

