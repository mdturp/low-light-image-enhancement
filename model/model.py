# The main pytorch model code is taken from
# https://github.com/vis-opt-group/SCI/blob/main/model.py
# and modified to contain only the parts that are needed for inference.
# The original code is licensed under MIT License.
import torch
import torch.nn as nn


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class Finetunemodel(nn.Module):
    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)

        base_weights = torch.load(weights, map_location=torch.device("cpu"))
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return r
