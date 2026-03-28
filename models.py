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
    # ... (Keep your existing UNet code here) ...
    def __init__(self, n_classes=16): # Changed to 16 for GID
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

class SwinDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample the lower resolution feature map
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Concat channels: (in_channels // 2) + skip_channels
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Concatenate along the channel dimension instead of addition
        x = torch.cat([x, skip], dim=1) 
        return self.conv(x)

class SwinUNet(nn.Module):
    def __init__(self, n_classes=16): # GID has 16 classes (0-15)
        super().__init__()
        # MUST BE TRUE for Transfer Learning
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True)
        
        # Swin-Tiny feature channels: f[0]=96, f[1]=192, f[2]=384, f[3]=768
        # Proper dense decoder blocks
        self.up1 = SwinDecoderBlock(in_channels=768, skip_channels=384, out_channels=384)
        self.up2 = SwinDecoderBlock(in_channels=384, skip_channels=192, out_channels=192)
        self.up3 = SwinDecoderBlock(in_channels=192, skip_channels=96, out_channels=96)
        
        self.final = nn.Sequential(
            nn.Conv2d(96, n_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear') # Upsample back to original 224x224
        )

    def forward(self, x):
        f = self.backbone(x) 
        # Fix: Rearrange from [B, H, W, C] to [B, C, H, W]
        f = [feat.permute(0, 3, 1, 2).contiguous() for feat in f]

        d1 = self.up1(f[3], f[2])
        d2 = self.up2(d1, f[1])
        d3 = self.up3(d2, f[0])
        
        return self.final(d3)