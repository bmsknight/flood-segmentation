import torch
import numpy as np

from src.dataset import SegDataset, SegDataLoader
from src.vision_transformer import SegFormer, Learner
from src.utils import check_accuracy

import src.constants as const

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training will happen on : ", device)

# setting seed for reproducibility
torch.manual_seed(const.RANDOM_SEED)
np.random.seed(const.RANDOM_SEED)

# load dataset
train_dataset = SegDataset("data/Train/Image","data/Train/Mask")
train_loader = SegDataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=const.SHUFFLE).get_dataloader()

test_dataset = SegDataset("data/Test/Image","data/Test/Mask")
test_loader = SegDataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False).get_dataloader()

# Set the network, loss and optimizer

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

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

model_trainer = Learner(network,loss,optimizer,20,device,model_save_path="models/best.pth")
history = model_trainer.train(train_loader, val_loader=test_loader)
model_trainer.load_best_model()
accuracy, dice_score, iou,map = check_accuracy(test_loader,model_trainer.network,device)
print(f'Accuracy:{accuracy}\nDice_Score:{dice_score}')
