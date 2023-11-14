import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

from networks.xception import TransferModel


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)

    def forward(self, x):
        x = self.net(x)
        return x


class MyEfficientNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(MyEfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def part1(self, inputs):
        return self._swish(self._bn0(self._conv_stem(inputs)))

    def part2(self, x):
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x

    def part3(self, inputs):
        return self._swish(self._bn1(self._conv_head(inputs)))

    def classifier(self, inputs):
        # Pooling and final linear layer
        feat = self._avg_pooling(inputs)
        feat = feat.flatten(start_dim=1)
        out = self._fc(self._dropout(feat))
        return out, feat

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        return self.classifier(x)



if __name__ == '__main__':
    inputs = torch.rand(2, 3, 256, 256)
    model = MyEfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
    print(model(inputs))
