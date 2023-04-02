import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
def dice_score(y_true, y_pred, smooth=1e-12):
    y_true_f = y_true.view(-1).float()
    y_pred_f = y_pred.view(-1).float()
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1e-12):
        inputs = F.sigmoid(inputs)
        inputs = (inputs > 0.5).float().requires_grad_()
        dc = dice_score(inputs, targets)
        return 1 - dc