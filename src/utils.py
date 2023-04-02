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

def evaluate(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0
    ap = 0
    model.eval()
    seen_samples = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(DEVICE)
            Y = Y.int().to(DEVICE)
            seen_samples += X.shape[0]

            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).int()
            num_correct += (preds == Y).sum()
            num_pixels += torch.numel(preds)
            dice_score += 2 *((preds & Y).sum((2, 3)) / ((preds+Y).sum((2, 3)) + 1e-12)).sum()
            iou += ((preds & Y).sum((2, 3)) / ((preds | Y).sum((2, 3)) + 1e-12)).sum()
            ap += sum(average_precision_score(y.ravel(), pred.ravel()) for y, pred in zip(Y, preds))
    accuracy =  num_correct*100.0/num_pixels
    dice_score = dice_score/ seen_samples
    iou =  iou/ seen_samples
    ap = ap/ seen_samples
    print(f"Accuracy: {num_correct}/{num_pixels} = {accuracy}%")
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

def visualize(dataloader, model_dice, model_bce=None):# displaying random image, mask, and predicted mask
    import random
    import matplotlib.pyplot as plt

    plt.figure(figsize=(30,30))
    sub_plots = 3  # 4
    ind = random.randint(0, len(dataloader.dataset))
    img, mask = dataloader.dataset[ind]
    predMask_dice = (torch.sigmoid(model_dice(img.unsqueeze(0)))> 0.5).float()
    # predMask_bce = (torch.sigmoid(model_bce(img.unsqueeze(0)))> 0.5).float()
    img = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
    mask = np.transpose(mask.detach().numpy(), (1, 2, 0))
    predMask_dice = np.transpose(predMask_dice.squeeze(0).detach().numpy(), (1, 2, 0))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, sub_plots, 1)
    ax1.set_title("original image", fontdict = {'fontsize' : 10})
    ax1.imshow(img)
    ax2 = fig.add_subplot(1, sub_plots, 2)
    ax2.set_title("original mask", fontdict = {'fontsize' : 10})
    ax2.imshow(np.squeeze(mask))
    ax3 = fig.add_subplot(1, sub_plots, 3)
    ax3.set_title("pred via U-Net", fontdict = {'fontsize' : 10})
    ax3.imshow(np.squeeze(predMask_dice))
    # ax4 = fig.add_subplot(1, sub_plots, 4)
    # ax4.set_title("pred via ENET", fontdict = {'fontsize' : 10})
    # ax4.imshow(np.squeeze(predMask_bce))
    plt.savefig('pred.png')
    print(f'saved {"pred.png"}')
