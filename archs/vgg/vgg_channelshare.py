from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

tau_global = 1. / (1. - 0.25)


class VGG16_channelshare(nn.Module):
    def __init__(
        self,  num_classes = 10, total_timestep = 5, num_chgroup=2
    ) -> None:
        super().__init__()
        self.total_timestep = total_timestep
        self.num_chgroup = num_chgroup


        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif1 =  neuron.LIFNode(v_threshold=1.0, v_reset=None, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.lif5 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.lif6 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.lif7 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.lif8 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.lif9 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.lif10 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.lif11 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.lif12 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.lif13 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=tau_global,
                                    surrogate_function=surrogate.ATan(),
                                    detach_reset=True)

        self.lastpool = nn.AdaptiveAvgPool2d((1,1))


        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def channel_divlif(self, input,  lif_module):
        N_list = []
        for i in range(self.num_chgroup):
            out_temp = lif_module(
                input[:, (input.size(1) // self.num_chgroup) * i:(input.size(1) // self.num_chgroup) * (i + 1), ...])
            N_list.append(out_temp)
        out = torch.cat(N_list, dim=1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output_list = []

        static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            out = self.channel_divlif(static_x, self.lif1)

            out = self.bn2(self.conv2(out))
            out = self.channel_divlif(out, self.lif2)
            out = self.maxpool1(out)

            out = self.bn3(self.conv3(out))
            out = self.channel_divlif(out, self.lif3)
            out = self.bn4(self.conv4(out))
            out = self.channel_divlif(out, self.lif4)
            out = self.maxpool2(out)

            out = self.bn5(self.conv5(out))
            out = self.channel_divlif(out, self.lif5)
            out = self.bn6(self.conv6(out))
            out = self.channel_divlif(out, self.lif6)
            out = self.bn7(self.conv7(out))
            out = self.channel_divlif(out, self.lif7)
            out = self.maxpool3(out)

            out = self.bn8(self.conv8(out))
            out = self.channel_divlif(out, self.lif8)
            out = self.bn9(self.conv9(out))
            out = self.channel_divlif(out, self.lif9)
            out = self.bn10(self.conv10(out))
            out = self.channel_divlif(out, self.lif10)
            out = self.maxpool4(out)

            out = self.bn11(self.conv11(out))
            out = self.channel_divlif(out, self.lif11)
            out = self.bn12(self.conv12(out))
            out = self.channel_divlif(out, self.lif12)
            out = self.bn13(self.conv13(out))
            out = self.channel_divlif(out, self.lif13)
            out = self.lastpool(out)

            out = torch.flatten(out, 1)
            out = self.classifier(out)
            output_list.append(out)

        return output_list

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

