import torch
import torchvision
from src.unet_model import DEVICE

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print(f"Saving checkpoint as : {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds+y).sum() + 1e-8)

    accuracy =  num_correct*100.0/num_pixels
    dice_score = dice_score/ len(loader)
    print(f"{num_correct}/{num_pixels} with accuracy {accuracy}%")
    print(f"Dice score: {dice_score}")

    return accuracy, dice_score


def save_predictions_as_images(loader, model, directory = "saved_images", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        # y = y.to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{directory}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{directory}/gt_{idx}.png"
        )
    
    model.train()