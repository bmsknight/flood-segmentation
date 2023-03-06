import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class SegDataset(Dataset):
    def __init__(self, 
                 image_dir,
                 mask_dir,
                 augment=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __get_item__(self, 
                     index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", ".png")) 
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32)  # Grey scale Mask
        mask = np.where(mask>127, 255., 0.) # Convert to binary values

        if self.augment:
            pass

        return image, mask


class SegDataLoader(DataLoader):
    def __init__(self, 
                 dataset,
                 batch_size=8,
                 shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        dataloader = DataLoader(self.dataset, self.batch_size, self.shuffle)
        return dataloader
    
def split_dataset(dataset, test_size, random_seed=42):
    """
    Split Pytorch Dataset for training and testing
    """
    test_size = int(dataset.__len__()*test_size) if test_size < 1.0 else int(test_size)
    train_size = dataset.__len__() - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    return train_dataset, test_dataset