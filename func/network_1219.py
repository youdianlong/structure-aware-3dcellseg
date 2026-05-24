import torch
import torch.nn as nn
import torch.nn.functional as F

# 12.19 innovation: 3D SE block and 3D ASPP (Atrous Spatial Pyramid Pooling)
# ==========================================
# Module 1: 3D Squeeze-and-Excitation (SE) Block
# Function: channel attention mechanism that suppresses background noise and emphasizes important features.
# ==========================================
class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


# ==========================================
# Module 2: 3D ASPP (Atrous Spatial Pyramid Pooling)
# Function: multi-scale perception that expands the receptive field and adapts to cells of different sizes.
# ==========================================
class ASPP3D(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[2, 4, 8]):
        super(ASPP3D, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv3d(in_dims, out_dims, 1, bias=False),
            nn.GroupNorm(8, out_dims),
            nn.ReLU(inplace=True)
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv3d(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0], bias=False),
            nn.GroupNorm(8, out_dims),
            nn.ReLU(inplace=True)
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv3d(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1], bias=False),
            nn.GroupNorm(8, out_dims),
            nn.ReLU(inplace=True)
        )
        self.aspp_block4 = nn.Sequential(
            nn.Conv3d(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2], bias=False),
            nn.GroupNorm(8, out_dims),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_dims, out_dims, 1, stride=1, bias=False),
            nn.GroupNorm(8, out_dims),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv3d(out_dims * 5, out_dims, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_dims)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# ==========================================
# Module 3: Basic convolution block with an integrated SE Block
# ==========================================
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock3D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


# ==========================================
# Main model: CellSegNet_1219 (SE-ASPP-UNet)
# ==========================================
class CellSegNet_1219(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_1219, self).__init__()

        # --- Encoder (downsampling) ---
        # Keep channel counts similar to the previous VoxResNet/UNet while adding SE.
        self.encoder1 = SingleConv(input_channel, 32, use_se=True)
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = SingleConv(32, 64, use_se=True)
        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = SingleConv(64, 128, use_se=True)
        self.pool3 = nn.MaxPool3d(2)

        # --- Bottleneck ---
        # Use ASPP instead of standard convolution to capture multi-scale features.
        # Input 128 -> output 256
        self.bottleneck_aspp = ASPP3D(128, 256)

        # --- Decoder (upsampling) ---
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = SingleConv(128 + 128, 128, use_se=False)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = SingleConv(64 + 64, 64, use_se=False)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = SingleConv(32 + 32, 32, use_se=False)

        # --- Output ---
        self.final_conv = nn.Conv3d(32, n_classes, kernel_size=1)
        self.output_func = output_func

    def forward(self, x, return_features=False):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        # Bottleneck (ASPP)
        bn = self.bottleneck_aspp(p3)

        # Decoder
        d3 = self.up3(bn)
        d3 = torch.cat((d3, e3), dim=1)  # Skip Connection
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.final_conv(d1)

        if self.output_func == "softmax":
            out = F.softmax(out, dim=1)

        # Compatible with the previous DANN training code by returning intermediate features for adversarial learning.
        if return_features:
            return out, [e1, e2, e3, bn]
        else:
            return out
