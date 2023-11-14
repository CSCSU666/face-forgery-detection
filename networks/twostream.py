import os

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from networks.efficientnet import MyEfficientNet
from networks.se_module import SELayer
from networks.xception import TransferModel
from timm.models.vision_transformer import VisionTransformer


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSSABlock(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[1, 3, 5, 7], stride=1, conv_groups=[1, 1, 1, 1]):
        super(MSSABlock, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(inplans, inplans, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inplans),
            nn.ReLU()
        )

        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.sa = SpatialAttention()
        self.softmax = nn.Softmax(dim=1)
        self.split_channel = planes // 4
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_x, input_y):
        x = self.convblk(torch.cat((input_x, input_y), dim=1))
        batch_size = x.shape[0]
        w, h = x.shape[2], x.shape[3]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_sa = self.sa(x1)
        x2_sa = self.sa(x2)
        x3_sa = self.sa(x3)
        x4_sa = self.sa(x4)

        x_sa = torch.cat((x1_sa, x2_sa, x3_sa, x4_sa), dim=1)
        attention_vectors = x_sa.view(batch_size, 4, 1, w, h)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        out1 = torch.cat((feats_weight[:, 0, :, :], feats_weight[:, 1, :, :]), dim=1)
        out2 = torch.cat((feats_weight[:, 2, :, :], feats_weight[:, 3, :, :]), dim=1)
        out1 = out1 + input_x
        out2 = out2 + input_y
        return self.relu(out1), self.relu(out2)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)


class EnhanceModule(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[1, 3, 5, 7], stride=1, conv_groups=[1, 1, 1, 1]):
        super(EnhanceModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.ca = ChannelAttention(planes)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.ca(out) * x
        return out


class TwoStream(nn.Module):
    def __init__(self):
        super(TwoStream, self).__init__()
        self.net_rgb = MyEfficientNet.from_pretrained("efficientnet-b0", advprop=True, num_classes=2)
        self.net_srm = MyEfficientNet.from_pretrained("efficientnet-b0", advprop=True, num_classes=2)

        self.num_features = 2048

        self.srm_conv = SRMConv2d_simple(inc=3)

        self.im = MSSABlock(64, 64)

        self.fusion = VisionTransformer(
            img_size=7,
            patch_size=7,
            in_chans=2560,
            num_classes=2,
            embed_dim=2048,
            depth=1,
            num_heads=4
        )

    def forward(self, x):
        y = self.srm_conv(x)

        x = self.net_rgb.part1(x)
        y = self.net_srm.part1(y)

        x, y = self.im(x, y)

        x = self.net_rgb.part2(x)
        y = self.net_srm.part2(y)

        x = self.net_rgb.part3(x)
        y = self.net_srm.part3(y)

        feat = self.fusion.forward_features(torch.cat((x, y), dim=1))
        out = self.fusion.head(feat)

        return out, feat


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    inputs = torch.rand(2, 3, 256, 256).cuda()
    model = TwoStream().cuda()
    print(model(inputs))
