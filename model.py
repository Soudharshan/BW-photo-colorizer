"""A small U-Net: grayscale L channel in, a/b colour channels out."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ColorizerNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = Block(1, base)
        self.enc2 = Block(base, base * 2)
        self.enc3 = Block(base * 2, base * 4)
        self.enc4 = Block(base * 4, base * 8)
        self.bottleneck = Block(base * 8, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.dec4 = Block(base * 16, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = Block(base * 8, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = Block(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = Block(base * 2, base)

        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        # x: (B, 1, H, W), H and W divisible by 16
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        b = self.bottleneck(self.pool(x4))

        d4 = self.dec4(torch.cat([self.up4(b), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        return torch.tanh(self.out(d1))   # ab in [-1, 1]
