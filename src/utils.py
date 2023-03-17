import sys
sys.path.append('./')
import os
import numpy as np
import torch
import torchvision
from src.unet_model import DEVICE
from sklearn.metrics import average_precision_score

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print(f"Saving checkpoint as : {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device=DEVICE):
    num_correct = []
    num_pixels = []
    dice_score = []
    iou = []
    ap = []
    model.eval()
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            Preds = torch.sigmoid(model(X))
            Preds = (Preds > 0.5).float()
            for pred, y in zip(Preds, Y):
                num_correct.append((pred == y).sum())
                num_pixels.append(torch.numel(pred))
                dice_score.append((2 * (pred * y).sum()) / (pred.sum() + y.sum() + 1e-12))
                iou.append((pred * y).sum() / ((torch.logical_or(pred, y)).sum() + 1e-12))
                ap.append(average_precision_score(y.ravel(), pred.ravel()))
    accuracy =  np.array(num_correct).sum()*100.0/np.array(num_pixels).sum()
    dice_score = np.array(dice_score).mean()
    iou =  np.array(iou).mean()
    ap = np.array(ap).mean()
    print(f"Accuracy: {accuracy}%")
    print(f"Dice score: {dice_score}")
    print(f"IoU: {iou}")
    print(f"AP: {ap}")
    return accuracy, dice_score, iou, ap

def save_predictions_as_images(loader, model, directory = "saved_images", device="cuda"):
    os.makedirs(directory, exist_ok=True)
    model.eval()
    for idx, (x_batch, y_batch) in enumerate(loader):
        for batch_idx, (x, y) in enumerate(zip(x_batch, y_batch)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                preds = torch.sigmoid(model(x.unsqueeze(0)))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{directory}/pred_{idx}_{batch_idx}.png"
            )
            torchvision.utils.save_image(
                y.unsqueeze(0), f"{directory}/gt_{idx}_{batch_idx}.png"
            )

def evaluate_all(model, X, Y):
    dice_scores = []
    iou_scores = []
    ap_scores = []

    for i in range(len(X)):
        img = X[i]
        predMask = model.predict(np.expand_dims(img, axis=0), verbose=0)
        #predMask = np.squeeze(predMask)
        trueMask = Y[i]

        # Dice score
        intersection = np.logical_and(trueMask, predMask)
        dice = (2. * intersection.sum()) / (trueMask.sum() + predMask.sum())
        dice_scores.append(dice)

        # IoU score
        intersection = np.logical_and(trueMask, predMask)
        union = np.logical_or(trueMask, predMask)
        iou = intersection.sum() / union.sum()
        iou_scores.append(iou)

        # Average precision score
        ap = average_precision_score(trueMask.ravel(), predMask.ravel())
        ap_scores.append(ap)

    # Compute mean scores
    mean_dice_score = np.mean(dice_scores)
    mean_iou_score = np.mean(iou_scores)
    mean_ap_score = np.mean(ap_scores)

    print(f"Mean Dice score: {mean_dice_score}")
    print(f"Mean IoU score: {mean_iou_score}")
    print(f"Mean Average Precision score: {mean_ap_score}")