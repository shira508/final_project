import torch
import torch.nn as nn


class InpaintingGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        # ======================
        # Encoder
        # ======================

        self.down1 = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ======================
        # Bottleneck
        # ======================

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ======================
        # Decoder
        # ======================

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256 + 256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128 + 128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64 + 64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, img, mask):

        x = torch.cat([img, mask], dim=1)

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder
        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)

        out = self.final(u1)

        return out
    


import torch
import torch.nn as nn


class InpaintingDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, img, mask):

        x = torch.cat([img, mask], dim=1)

        return self.model(x)   