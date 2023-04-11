import sys
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.append('./')
import os
import torch
import torchvision
from torchmetrics.classification import BinaryJaccardIndex
# from src.unet_model import DEVICE

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print(f"Saving checkpoint as : {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = []
    num_pixels = []
    dice_score = []
    iou = []
    ap = []

    jack = BinaryJaccardIndex().to(device)
    iou1 = 0
    model.eval()

    with torch.no_grad():
        for x, Y in loader:
            x = x.to(device)
            Y = Y.to(device)

            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            Y = torch.squeeze(Y,dim=1)

            for pred, y in zip(preds, Y):
                pred = pred.cpu()
                y = y.cpu()
                num_correct.append((pred == y).sum())
                num_pixels.append(torch.numel(pred))
                dice_score.append((2 * (pred * y).sum()) / (pred.sum() + y.sum() + 1e-12))
                iou.append((pred * y).sum() / ((torch.logical_or(pred, y)).sum() + 1e-12))
                ap.append(average_precision_score(y.ravel(), pred.ravel()))

            iou1 += jack(preds,Y)

    accuracy = np.array(num_correct).sum() * 100.0 / np.array(num_pixels).sum()
    dice_score = np.array(dice_score).mean()
    iou = np.array(iou).mean()
    ap = np.array(ap).mean()

    iou1 = iou1/len(loader)

    print(f"Accuracy: {accuracy}%")
    print(f"Dice score: {dice_score}")
    print(f"IoU: {iou}")
    print(f"AP: {ap}")

    print(f"mIoU: {iou1}")

    return accuracy, dice_score, iou, ap


def save_predictions_as_images(loader, model, directory = "saved_images", device="cuda"):
    os.makedirs(directory, exist_ok=True)
    model.eval()
    for idx, (x_batch, y_batch) in enumerate(loader):
        for batch_idx, (x, y) in enumerate(zip(x_batch, y_batch)):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x.unsqueeze(0)))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{directory}/pred_{idx}_{batch_idx}.png"
            )
            torchvision.utils.save_image(
                y.unsqueeze(0), f"{directory}/gt_{idx}_{batch_idx}.png"
            )