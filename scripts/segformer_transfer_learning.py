import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from src.segformer_reprogram import SemanticSegmentationDataset

feature_extractor = SegformerFeatureExtractor(reduce_labels=False, random_crop=False)

train_dataset = SemanticSegmentationDataset(root_dir="../data/Train", feature_extractor=feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = SemanticSegmentationDataset(root_dir="../data/Test", feature_extractor=feature_extractor)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

id2label = {0: 'bg', 1: 'flood'}
label2id = {'bg': 0, 'flood': 1}

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=2,
                                                         id2label=id2label,
                                                         label2id=label2id)

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

model.train()
for epoch in range(0, 50):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

        # let's print loss and metrics every 100 batches
        if idx % 50 == 0:
            print("Loss:", loss.item())
    with torch.no_grad():
        dice_score = []
        iou = []
        ap = []
        for idx, batch in enumerate(tqdm(test_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            for pred, y in zip(predicted, labels):
                pred = pred.cpu()
                y = y.cpu()
                dice_score.append((2 * (pred * y).sum()) / (pred.sum() + y.sum() + 1e-12))
                iou.append((pred * y).sum() / ((torch.logical_or(pred, y)).sum() + 1e-12))
                ap.append(average_precision_score(y.ravel(), pred.ravel()))

        dice_score = np.array(dice_score).mean()
        iou = np.array(iou).mean()
        ap = np.array(ap).mean()
        print(f"Dice score: {dice_score}")
        print(f"IoU: {iou}")
        print(f"AP: {ap}")
        model.save_pretrained(f"../models/segformer_{epoch}pth")
