import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch
from torch.autograd import Function
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

        self.pool = nn.AdaptiveAvgPool3d(1)

        # Apply spectral_norm only to Linear layers.
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_features, 32)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(32, num_domains))
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
