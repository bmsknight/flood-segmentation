import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        self.img_dir = os.path.join(self.root_dir, "Image")
        self.ann_dir = os.path.join(self.root_dir, "Mask")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            print(files)
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert('RGB')
        annotation = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        annotation = np.array(annotation.convert("L")).astype(np.float32)
        annotation_2d = (annotation > 127).astype(np.uint8)

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(image, Image.fromarray(annotation_2d), return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class InputTransformLayer(nn.Module):
    def __init__(self, width, height, channels):
        super().__init__()
        # we want the input to be as close as possible to original
        # hence weight is initiated with ones bias initiated with 0
        # but they will obviously change over time with learning
        self.weight = nn.Parameter(torch.ones(channels, width, height))
        self.bias = nn.Parameter(torch.zeros(channels, width, height))

    def forward(self, x):
        out = self.weight * x + self.bias
        return out


class SegformerReprogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = InputTransformLayer(512, 512, 3)
        self.seg_former = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
                                                                           )
        for p in self.seg_former.parameters():
            p.requires_grad = False
        self.act = nn.ReLU()
        self.output_transform = nn.Conv2d(150, 2, kernel_size=1)

    def forward(self, x):
        out = self.input_transform(x)
        out = self.seg_former(out)
        out = out.logits
        # out = self.act(out)
        out = self.output_transform(out)
        return out
