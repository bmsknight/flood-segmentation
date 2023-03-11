"Inspired by https://github.com/CodeProcessor/Unet-PyTorch-Implementation/blob/master/model.py"

import torch
import torch.nn as nn
from torchsummary import summary
from src.constants import *
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UnetModel(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=[64, 128, 256, 512]) -> None:
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=1, padding=1))
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x) 
            skip_connections.append(x)
            x = self.pool(x)
        
        skip_connections = skip_connections[::-1] 
        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_con = skip_connections[idx//2]

            if x.shape != skip_con.shape:
                x = TF.resize(x, size=skip_con.shape[2:])

            concat_x = torch.cat((skip_con, x), dim=1)
            x = self.ups[idx+1](concat_x)

        return self.final_conv(x)


class DoubleConv(nn.Module):
    """
    Reusable Double convolutional 
    """
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def test():
    input_shape = (IMG_H, IMG_W)
    model = UnetModel(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    summary(model, input_size=(3, input_shape[0], input_shape[1]), device=DEVICE)
    x = torch.randn(2, 3, input_shape[0], input_shape[1], device=DEVICE)
    print(model(x).shape)


if __name__ == '__main__':
    test()