import torch
import torch.nn as nn
import torch.nn.functional as F


class VAPEN(nn.Module):
    def __init__(self, iter_num=5):
        super(VAPEN, self).__init__()
        self.act = nn.Sigmoid()
        number_f = 16
        self.conv1_1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.conv1_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv2_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv2_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv4_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv4_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv5_1 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv5_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv6_1 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv6_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv7_1 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv7_2 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.iter_num = iter_num

    def forward(self, x):
        x0_1 = self.conv1_1(x)
        x0_2 = self.act(self.conv1_2(x0_1))

        # en 1_1
        x1_1 = self.pool(x0_2)
        x1_1 = self.conv2_1(x1_1)
        # en 1_2
        x1_2 = self.act(self.conv2_2(x1_1))

        # en 2_1
        x2_1 = self.pool(x1_2)
        x2_1 = self.conv3_1(x2_1)
        # en 2_2
        x2_2 = self.act(self.conv3_2(x2_1))

        # en 3_1
        x3_1 = self.pool(x2_2)
        x3_1 = self.conv4_1(x3_1)
        # en 3_2
        x3_2 = self.act(self.conv4_2(x3_1))

        # de 1_1
        x4_1 = self.up(x3_2)
        x4_1 = self.conv5_1(torch.cat([x2_2, x4_1], 1))
        # de 1_2
        x4_2 = self.act(self.conv5_2(x4_1))

        # de 2_1
        x5_1 = self.up(x4_2)
        x5_1 = self.conv6_1(torch.cat([x1_2, x5_1], 1))
        # de 2_2
        x5_2 = self.act(self.conv6_2(x5_1))

        # de 3_1
        x6_1 = self.up(x5_2)
        x6_1 = self.conv7_1(torch.cat([x0_2, x6_1], 1))
        # de 3_2
        x_r = torch.tanh(self.conv7_2(x6_1))

        for i in range(self.iter_num):
            # x = torch.pow(x, 2) * x_r * (x + 1) + x * (1 - 2 * x_r)
            x = x_r * (torch.pow(x, 2) - 1) * x + x

        return x, x_r
