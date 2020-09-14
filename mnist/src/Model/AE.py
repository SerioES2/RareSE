import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, image_size=64):
        super(AutoEncoder, self).__init__()

        self.encLayer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size).
            nn.ReLU(inplace=True)
        )

        self.encLayer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True)
        )

        self.encLayer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True)
        )

        self.encLayer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True)
        )
        return

    def forward(self, x):
        out = self.encLayer1(x)
        out = self.encLayer2(out)
        out = self.encLayer3(out)
        out = self.encLayer4(out)
        return out