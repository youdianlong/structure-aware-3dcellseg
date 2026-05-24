import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient.
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return GradientReversalFunc.apply(input, self.alpha)




class voxel_DomainDiscriminator(nn.Module):
    def __init__(self, in_features=64, num_domains=1): # Note: num_domains is changed to 1 for binary logits.
        super().__init__()

        # The original code used AdaptiveAvgPool3d(1) -> Flatten -> Linear.
        # The current code uses Conv3d throughout to preserve spatial structure.

        self.model = nn.Sequential(
            # First layer: feature extraction, preserving resolution or slightly downsampling.
            # Assume input c4 is (B, 64, 8, 8, 8) -> output (B, 64, 8, 8, 8).
            nn.Conv3d(in_features, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer: increase channels and extract deeper local features.
            # Output: (B, 128, 8, 8, 8)
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), # Adding normalization is recommended for stable training.
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer: output layer, namely the PatchGAN output.
            # Use a 1x1x1 convolution to reduce channels to 1.
            # Output (B, 1, 8, 8, 8) represents discriminator results for 8x8x8 local regions.
            nn.Conv3d(128, num_domains, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # x: (B, 64, D, H, W)
        # Output: (B, 1, D, H, W)
        return self.model(x)
