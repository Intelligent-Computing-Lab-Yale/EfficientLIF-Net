from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

tau_global = 1. / (1. - 0.25)


class VGG16(nn.Module):
    def __init__(
        self,  num_classes: int = 10, total_timestep: int = 5
    ) -> None:
        super().__init__()
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif1 =  neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.lif5 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.lif6 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.lif7 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.lif8 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.lif9 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.lif10 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.lif11 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.lif12 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.lif13 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                    surrogate_function=surrogate.ATan(),
                                    detach_reset=True)

        self.lastpool = nn.AdaptiveAvgPool2d((1,1))


        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output_list = []

        static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            out  =self.lif1(static_x)
            out = self.lif2(self.bn2(self.conv2(out)))
            out = self.maxpool1(out)

            out = self.lif3(self.bn3(self.conv3(out)))
            out = self.lif4(self.bn4(self.conv4(out)))
            out = self.maxpool2(out)

            out = self.lif5(self.bn5(self.conv5(out)))
            out = self.lif6(self.bn6(self.conv6(out)))
            out = self.lif7(self.bn7(self.conv7(out)))
            out = self.maxpool3(out)

            out = self.lif8(self.bn8(self.conv8(out)))
            out = self.lif9(self.bn9(self.conv9(out)))
            out = self.lif10(self.bn10(self.conv10(out)))
            out = self.maxpool4(out)

            out = self.lif11(self.bn11(self.conv11(out)))
            out = self.lif12(self.bn12(self.conv12(out)))
            out = self.lif13(self.bn13(self.conv13(out)))
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

