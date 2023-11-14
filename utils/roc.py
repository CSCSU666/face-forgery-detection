import numpy as np

from scipy import interpolate

from sklearn.metrics import roc_curve


def compute_accuracy(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / (len(pred_idx) + 0.5)


def compute_eer(predicted, target):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(fpr, tpr)
    y = function(scale)
    znew = abs(scale + y - 1)
    eer = scale[np.argmin(znew)]
    return eer
