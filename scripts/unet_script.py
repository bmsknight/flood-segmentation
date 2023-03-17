"Inspired by https://github.com/CodeProcessor/Unet-PyTorch-Implementation/blob/master/model.py"
import sys
sys.path.append('./')

import os
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms.functional as TF

from src.dataset import SegDataset, SegDataLoader
from src.unet_model import UnetModel
from src.utils import *
from src.constants import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT_NAME = datetime.datetime.now().strftime("%Y-%b-%d_%Hh-%Mm-%Ss_") + "experiment_1"


class UNET:
    def __init__(self,
                 train_folder=None,
                 test_folder=None,
                 load_weights_path=None,
                 save_weights_path=os.path.join('runs',f'{EXPERIMENT_NAME}','checkpoints')) -> None:
        
        self.load_weights_path = load_weights_path
        self.save_weights_path = save_weights_path
        self.model = self.build_model(IN_CHANNELS, OUT_CHANNELS)

        if self.load_weights_path is not None:
            load_checkpoint(torch.load(self.load_weights_path), self.model)

        if train_folder is not None:
            self.train_data_loader = self.init_data_loader(train_folder)

        if test_folder is not None:
            self.test_data_loader = self.init_data_loader(test_folder)

    def build_model(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        return UnetModel(in_channels, out_channels, features)
    
    def init_data_loader(self, data_folder):
        dataset = SegDataset(os.path.join(data_folder, 'Image'), os.path.join(data_folder, 'Mask'))
        data_loader = SegDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE).get_dataloader()
        return data_loader

    def train_fn(self, optimizer, loss_fn, scaler):
        print('Training in progress...')
        loop = tqdm(self.train_data_loader)

        for batch_idx, (data, targets) in enumerate(self.train_data_loader):
            data = data.to(device=DEVICE)
            targets = targets.float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    def train(self):
        loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scalar = torch.cuda.amp.GradScaler()

        for epoch in range(START_EPOCH, EPOCHS):
            print(f"Epoch: {epoch}")

            self.train_fn(optimizer, loss_fn, scalar)

            # Save model
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            os.makedirs(self.save_weights_path, exist_ok=True)
            save_checkpoint(checkpoint, filename=os.path.join(self.save_weights_path, f"unet_checkpoint_{epoch}.pth"))

            self.test(save_preds=True, directory=os.path.join('saved_images', f'{EXPERIMENT_NAME}', f'epoch-{epoch}_saved_images'))

    def test(self, directory=os.path.join('runs',f'{EXPERIMENT_NAME}','saved_images'), save_preds=False):
                # check accuracy
                _ = check_accuracy(self.test_data_loader, self.model, device=DEVICE)
                evaluate_all(self.model, self.test_data_loader)
                if save_preds:
                    print('Saving predictions...')
                    # print some examples to a folder
                    save_predictions_as_images(self.test_data_loader, self.model, device=DEVICE, directory=directory)


if __name__ == '__main__':
    unet = UNET(TRAIN_DATA_DIR,
                TEST_DATA_DIR,
                load_weights_path='runs\\2023-Mar-15_08h-10m-03s\\checkpoints\\unet_checkpoint_7.pth')
    unet.test(save_preds=True)
    # unet.train()
    