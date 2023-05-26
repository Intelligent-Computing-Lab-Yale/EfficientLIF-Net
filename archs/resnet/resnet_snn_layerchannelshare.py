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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, info_list ):
        x, lif_module =info_list

        act = (self.bn1(self.conv1(x)))

        out1 = lif_module(act[:,:(act.size(1)//2),...])
        out2 = lif_module(act[:,(act.size(1)//2):,...])
        out = torch.cat([out1, out2], dim=1)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        act = out
        out1 = lif_module(act[:, :(act.size(1) // 2), ...])
        out2 = lif_module(act[:, (act.size(1) // 2):, ...])
        out = torch.cat([out1, out2], dim=1)

        return [out, lif_module]





class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10, total_timestep=6):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(128)

        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)

        self.layer1_lif = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)
        self.layer2_lif = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                         surrogate_function=surrogate.ATan(),
                                         detach_reset=True)
        self.layer3_lif = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                         surrogate_function=surrogate.ATan(),
                                         detach_reset=True)
        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_img):

        output_list = []
        static_x = self.bn1(self.conv1(input_img))

        for t in range(self.total_timestep):

            N_list = []
            for i in range(2):
                out_temp = self.lif_input(static_x[:, (static_x.size(1) // 2) * i:(static_x.size(
                    1) // 2) * (i + 1), ...])
                N_list.append(out_temp)
            out = torch.cat(N_list, dim=1)

            out = self.layer1([out, self.layer1_lif])
            out, _ = out
            out = self.layer2([out, self.layer2_lif])
            out, _ = out
            out = self.layer3([out, self.layer3_lif])
            out, _ = out
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)

            output_list.append(out)

        return output_list


def ResNet19_layerchannelshare(args, num_classes, total_timestep):
    return ResNet(args, BasicBlock, [3, 3, 2], num_classes, total_timestep)

