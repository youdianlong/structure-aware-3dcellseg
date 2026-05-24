import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class ResModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1):
        super(ResModule, self).__init__()
        self.batchnorm_module=nn.BatchNorm3d(num_features=in_channels)
        self.conv_module=nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        h=F.relu(self.batchnorm_module(x))
        h=self.conv_module(h)
        return h+x


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Reuse the existing ResModule.
class ResModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1):
        super(ResModule, self).__init__()
        self.batchnorm_module = nn.BatchNorm3d(num_features=in_channels)
        self.conv_module = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=padding, dilation=dilation)

    def forward(self, x):
        h = F.relu(self.batchnorm_module(x))
        h = self.conv_module(h)
        return h + x


class CellSegNet_MultiScale(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_MultiScale, self).__init__()

        # --- Original encoder section ---
        self.conv1 = nn.Conv3d(input_channel, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(32)

        # Stage 1: Downsample to 1/2
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)

        # New multi-scale fusion layer for Stage 1.
        # Receive the half-resolution raw image, process it with convolution, and concatenate it to the backbone.
        self.aux_conv_x2 = nn.Sequential(
            nn.Conv3d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # Fusion layer: backbone (64) + auxiliary (16) -> fused (64).
        self.fusion_x2 = nn.Conv3d(64 + 16, 64, kernel_size=1)

        self.resmodule1 = ResModule(64, 64)

        # Stage 2: Downsample to 1/4
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)

        # New multi-scale fusion layer for Stage 2.
        # Receive the quarter-resolution raw image.
        self.aux_conv_x4 = nn.Sequential(
            nn.Conv3d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # Fusion layer: backbone (64) + auxiliary (16) -> fused (64).
        self.fusion_x4 = nn.Conv3d(64 + 16, 64, kernel_size=1)

        self.resmodule2 = ResModule(64, 64)

        # Stage 3: Downsample to 1/8. This is usually too small, so no raw image is added here.
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)

        # --- Decoder section, unchanged ---
        self.deconv1 = nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, n_classes, kernel_size=3, stride=1, padding=1)

        self.output_func = output_func

    def _resize_input(self, x, scale_factor):
        """Helper function: downsample the 3D input image."""
        return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False)

    def forward(self, x, domain=False):
        # --- 1. Prepare multi-scale inputs ---
        # x: (B, C, D, H, W)
        x_half = self._resize_input(x, 0.5)  # 1/2 resolution
        x_quarter = self._resize_input(x, 0.25)  # 1/4 resolution

        # --- 2. Encoder forward pass ---
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))  # Full Res Feature

        # >>> Stage 1 (1/2 Res) >>>
        h = self.conv3(c1)  # (B, 64, D/2, H/2, W/2)

        # Innovation: inject raw-image features at the 1/2 scale.
        aux_feat_2 = self.aux_conv_x2(x_half)  # (B, 16, D/2, H/2, W/2)
        h = torch.cat([h, aux_feat_2], dim=1)  # Concat -> (B, 80, ...)
        h = self.fusion_x2(h)  # Fuse 1x1 -> (B, 64, ...)

        c2 = self.resmodule1(h)

        # >>> Stage 2 (1/4 Res) >>>
        h = self.conv4(c2)  # (B, 64, D/4, H/4, W/4)

        # Innovation: inject raw-image features at the 1/4 scale.
        aux_feat_4 = self.aux_conv_x4(x_quarter)
        h = torch.cat([h, aux_feat_4], dim=1)
        h = self.fusion_x4(h)

        c3 = self.resmodule2(h)

        # >>> Stage 3 (1/8 Res) >>>
        h = self.conv5(c3)
        c4 = self.resmodule3(h)

        # --- 3. Decoder forward pass, matching the original logic except for crop alignment ---

        # Deconv 1
        c4_up = self.deconv1(c4)
        c4_up = F.relu(self.bnorm2(c4_up))
        # Crop-align c4_up and c3.
        c4_up = self._crop_concat(c4_up, c3)
        h = c4_up + c3

        # Deconv 2
        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        # Crop-align c2_2 and c2.
        c2_2 = self._crop_concat(c2_2, c2)
        h = c2_2 + c2

        # Deconv 3
        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        # Crop-align c1_2 and c1.
        c1_2 = self._crop_concat(c1_2, c1)
        h = c1_2 + c1

        h = self.conv6(h)

        output = F.softmax(h, dim=1)

        if domain:
            return output, c4
        else:
            return output

    def _crop_concat(self, upsampled, bypass):
        """Helper function: handle size mismatches in U-Net skip connections."""
        # upsampled comes from deconvolution, while bypass comes from the encoder.
        # Usually upsampled is slightly larger than bypass or the same size, so crop its center.
        d_diff = upsampled.size(2) - bypass.size(2)
        h_diff = upsampled.size(3) - bypass.size(3)
        w_diff = upsampled.size(4) - bypass.size(4)

        # Simple center crop.
        # This assumes diff >= 0, meaning the upsampled size is at least the encoder feature size.
        # If the data size is irregular, more complex padding logic may be needed.
        d_start = d_diff // 2
        h_start = h_diff // 2
        w_start = w_diff // 2

        return upsampled[:, :,
               d_start: d_start + bypass.size(2),
               h_start: h_start + bypass.size(3),
               w_start: w_start + bypass.size(4)]
    
class VoxResNet(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func = "softmax"):
        super(VoxResNet, self).__init__()
        
        self.conv1a=nn.Conv3d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1a=nn.BatchNorm3d(num_features=32)
        self.conv1b=nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1b=nn.BatchNorm3d(num_features=32)
        self.conv1c=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res2=ResModule(64, 64)
        self.res3=ResModule(64, 64)
        self.bnorm3=nn.BatchNorm3d(num_features=64)
        self.conv4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res5=ResModule(64, 64)
        self.res6=ResModule(64, 64)
        self.bnorm6=nn.BatchNorm3d(num_features=64)
        self.conv7=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res8=ResModule(64, 64)
        self.res9=ResModule(64, 64)
        
        self.c1deconv=nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.c1conv=nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, padding=1)
        self.c2deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.c2conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c3deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=6, stride=4, padding=1)
        self.c3conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c4deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=10, stride=8, padding=1)
        self.c4conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        
        self.output_func = output_func
    def forward(self, x):
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = F.relu6(self.c1deconv(h))
        c1 = self.c1conv(c1)
        
        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.res2(h)
        h = self.res3(h)
        c2 = F.relu6(self.c2deconv(h))
        c2 = self.c2conv(c2)
        
        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.res5(h)
        h = self.res6(h)
        c3 = F.relu6(self.c3deconv(h))
        c3 = self.c3conv(c3)
        
        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.res8(h)
        h = self.res9(h)
        c4 = F.relu6(self.c4deconv(h))
        c4 = self.c4conv(c4)
        
        c = c1 + c2 + c3 + c4
        
        output = F.softmax(c, dim=1)
        
        return output
