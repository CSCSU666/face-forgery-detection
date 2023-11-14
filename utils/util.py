import json

from efficientnet_pytorch import EfficientNet

from networks.efficientnet import MyEfficientNet, Detector
from networks.enhance_xcep import EnhanceXcep
from networks.xception import TransferModel


def read_json(path):
    with open(path, mode="r") as f:
        d = json.load(f)
    return d


def write_json(w_dict, path):
    with open(path, 'w') as f:
        json.dump(w_dict, f)


def load_model(model_name, device):
    if model_name == 'xception':
        model = TransferModel(modelchoice='xception', num_out_classes=2, dropout=0.2).model
    elif model_name.startswith('efficientnet'):
        model = MyEfficientNet.from_pretrained(model_name, advprop=True, num_classes=2)
    elif model_name == 'enhancexcep':
        model = EnhanceXcep()
    elif model_name == 'en-b4':
        model = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
    elif model_name == 'sbi':
        model = Detector()
    else:
        raise Exception('Error model!')
    return model.to(device)


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
