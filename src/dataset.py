import sys
sys.path.append('./')
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from src.constants import IMG_H, IMG_W, BATCH_SIZE, SHUFFLE
torch.manual_seed(0)


class SegDataset(Dataset):
    def __init__(self, 
                 image_dir,
                 mask_dir,
                 img_h=IMG_H,
                 img_w=IMG_W,
                 transform=None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_h = img_h
        self.img_w = img_w
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([TF.to_tensor, 
                                                 transforms.Resize(size=(self.img_h, self.img_w))])
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", ".png")) 
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32)  # Grey scale Mask

        image = self.transform(image)
        mask = self.transform(mask)
        print(mask.dtype)

        return image, (mask>127).float()


class SegDataLoader:
    def __init__(self, 
                 dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=SHUFFLE) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        dataloader = DataLoader(self.dataset, self.batch_size, self.shuffle)
        return dataloader
    

class AugDataset:
    def __init__(self,
                 image_dir,
                 mask_dir,
                 img_h=IMG_H,
                 img_w=IMG_W,
                 augment=False) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment

        transform = self.get_transforms()
        self.dataset = SegDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    
    def get_transforms(self):
        transformations = [TF.to_tensor,
                           transforms.Resize(size=(self.img_h, self.img_w))]
        if self.augment:
            transformations.pop()
            transformations.extend([transforms.Resize(size=(2*self.img_h, 2*self.img_w)), 
                                    transforms.TenCrop((self.img_h, self.img_w)), 
                                    lambda x: torch.stack(list(x), dim=0)])
        data_transformation = transforms.Compose(transformations)
        return data_transformation

    def save_transform(self, dataset_dir):
        """
        Save transformed datasets
        """
        img_folder = os.path.join(dataset_dir, 'Image')
        mask_folder = os.path.join(dataset_dir, 'Mask')
        os.makedirs(img_folder, exist_ok=True) 
        os.makedirs(mask_folder, exist_ok=True) 
        for i_img, sample_batch in enumerate(self.dataset):
            if self.augment:
                img_batch, mask_batch = sample_batch
                for i_crop, (image, mask) in enumerate(zip(img_batch, mask_batch)):
                    img_name = f'img{i_img}_crop{i_crop}.jpg'
                    mask_name = f'img{i_img}_crop{i_crop}.png'
                    self.save_image(image, img_folder, img_name)
                    self.save_image(mask, mask_folder, mask_name)
            else:
                image, mask = sample_batch
                img_name = f'img{i_img}.jpg'
                mask_name = f'img{i_img}.png'
                self.save_image(image, img_folder, img_name)
                self.save_image(mask, mask_folder, mask_name)
    
    @staticmethod
    def save_image(image, folder, name):
        save_path = os.path.join(folder, name)
        print(f'Saved {name}!')
        utils.save_image(image, save_path)


if __name__=='__main__':
    aug_train_dataset = AugDataset(
        image_dir='data\\train\\Image',
        mask_dir='data\\train\\Mask',
        img_h=256,
        img_w=256,
        augment=True
    )

    aug_test_dataset = AugDataset(
        image_dir='data\\test\\Image',
        mask_dir='data\\test\\Mask',
        img_h=256,
        img_w=256,
        augment=False
    )

    aug_train_dataset.save_transform('dataset\\train')
    aug_test_dataset.save_transform('dataset\\test')

    # image_dir='data\\train\\Image'
    # mask_dir='data\\train\\Mask'
    # dataset = SegDataset(image_dir=image_dir, mask_dir=mask_dir)
    # dataloader = SegDataLoader(dataset=dataset, batch_size=8, shuffle=True).get_dataloader()
    # for index, data in enumerate(dataloader):
    #     print(index, data[0].shape, data[1].shape)
    #     print(torch.unique(data[1]))
    #     if index>2:
    #         break