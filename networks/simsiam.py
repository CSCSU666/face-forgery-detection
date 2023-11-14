import torch
import torch.nn as nn

from networks.efficientnet import MyEfficientNet
from networks.enhance_xcep import EnhanceXcep
from networks.twostream import TwoStream
from networks.xception import TransferModel


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if base_encoder == 'xception':
            self.encoder = TransferModel(modelchoice='xception', num_out_classes=2, dropout=0.5).model
            prev_dim = 2048
        elif base_encoder.startswith('efficientnet'):
            self.encoder = MyEfficientNet.from_pretrained(base_encoder, advprop=True, num_classes=2)
            prev_dim = self.encoder.num_features
            # build a 3-layer projector
        elif base_encoder == 'enhancexcep':
            self.encoder = EnhanceXcep()
            prev_dim = 2048
        else:
            raise Exception('Error model!')
        self.encoder.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                               nn.BatchNorm1d(prev_dim),
                                               nn.ReLU(inplace=True),  # first layer
                                               nn.Linear(prev_dim, prev_dim, bias=False),
                                               nn.BatchNorm1d(prev_dim),
                                               nn.ReLU(inplace=True),  # second layer
                                               nn.Linear(prev_dim, dim),
                                               nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        out1, feat1 = self.encoder(x1)  # NxC
        out2, feat2 = self.encoder(x2)  # NxC

        z1 = self.encoder.projector(feat1)
        z2 = self.encoder.projector(feat2)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return out1, out2, p1, p2, z1.detach(), z2.detach()


if __name__ == '__main__':
    x1 = torch.rand(2, 3, 299, 299)
    x2 = torch.rand(2, 3, 299, 299)
    model = SimSiam('xception')
    out = model(x1, x2)
    print('done')
