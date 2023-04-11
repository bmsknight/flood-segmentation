import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torchvision.transforms.functional as TF

from src.vision_transformer import SegFormer
from src.segformer_reprogram import SegformerReprogram

feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
id2label = {0: 'bg', 1: 'flood'}
label2id = {'bg': 0, 'flood': 1}

model = SegformerForSemanticSegmentation.from_pretrained("../models/segformer_46pth",
                                                         num_labels=2,
                                                         id2label=id2label,
                                                         label2id=label2id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

im_id = 3097

image = Image.open(f'../data/Test/Image/{im_id}.jpg')
mask = Image.open(f'../data/Test/Mask/{im_id}.png')

# Pretrained
# prepare the image for the model
encoding = feature_extractor(image, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)
# forward pass
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()
id2color = {0: [0, 0, 255], 1: [255, 0, 0]}

# First, rescale logits to original image size
upsampled_logits = nn.functional.interpolate(logits,
                                             size=(256, 256),  # (height, width)
                                             mode='bilinear',
                                             align_corners=False)
# Second, apply argmax on the class dimension
seg = upsampled_logits.argmax(dim=1)[0]
del model
# //////////////////////////////////////////////////////////////////////////////////////////

re_model = SegformerReprogram()
re_model.to(device)
re_model.load_state_dict(torch.load("../models/segformer_reprogram_v2_74.pth"))
re_output = re_model(pixel_values)
re_output = re_output.cpu()
re_upsample = nn.functional.interpolate(re_output,
                                             size=(256, 256),  # (height, width)
                                             mode='bilinear',
                                             align_corners=False)
re_seg = re_upsample.argmax(dim=1)[0]
del re_model
# /////////////////////////////////////////////////////////////////////////////////////////

network = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=2,
)
network.load_state_dict(torch.load("../models/best.pth"))
network.to(device)
T = transforms.Compose([TF.to_tensor, transforms.Resize(size=(256, 256))])
image_for_model = np.array(image.convert("RGB"))
image_for_model = T(image_for_model)
image_for_model = image_for_model.unsqueeze(0)
image_for_model = image_for_model.to(device)
preds = network(image_for_model)
preds = nn.functional.interpolate(preds,
                                  size=(256, 256),  # (height, width)
                                  mode='bilinear',
                                  align_corners=False)
preds = torch.argmax(preds, dim=1)
preds = preds.cpu().squeeze()

image = image.resize((256, 256), resample=1)
mask = mask.resize((256, 256), resample=1)

fig = plt.figure()
ax1 = fig.add_subplot(1, 5, 1)
ax1.set_title("original image", fontdict={'fontsize': 8})
ax1.imshow(image)
ax2 = fig.add_subplot(1, 5, 2)
ax2.set_title("original mask", fontdict={'fontsize': 8})
ax2.imshow(mask)
ax3 = fig.add_subplot(1, 5, 3)
ax3.set_title("Segformer ", fontdict={'fontsize': 8})
ax3.imshow(preds)
ax4 = fig.add_subplot(1, 5, 4)
ax4.set_title("Transfer Learning", fontdict={'fontsize': 8})
ax4.imshow(seg)
ax5 = fig.add_subplot(1, 5, 5)
ax5.set_title("Reprogramming", fontdict={'fontsize': 8})
ax5.imshow(re_seg)

plt.savefig(f"../output/{im_id}_reprog_v2.png")
