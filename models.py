import torch
import torch.nn as nn
import timm

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.final = nn.Conv2d(128, n_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        return self.final(torch.cat([d1, e1], dim=1))

class SwinUNet(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)
        self.up1 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.final = nn.Sequential(nn.Conv2d(96, n_classes, 1), nn.Upsample(scale_factor=4, mode='bilinear'))
    def forward(self, x):
        f = self.backbone(x)
        d1 = self.up1(f[3])
        d2 = self.up2(d1 + f[2])
        d3 = self.up3(d2 + f[1])
        return self.final(d3)