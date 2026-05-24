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




class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=64, num_domains=2):
        super(DomainDiscriminator, self).__init__()

        # Assume the c4 features from CellSegNet_basic_lite are used, with 64 channels.
        # Use adaptive average pooling to convert (B, 64, D, H, W) to (B, 64, 1, 1, 1).
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(True),
            nn.Linear(32, num_domains)  # Output logits for 2 classes.
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten to (B, 64).
        x = self.fc(x)
        return x
