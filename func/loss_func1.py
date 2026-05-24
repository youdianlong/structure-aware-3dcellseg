import torch
import numpy as np

def softmax_cross_entropy_loss(input, target, weights=None):
    """
    input: (B, C, D, H, W) - probability map after Softmax
    target: (B, C, D, H, W) - one-hot encoded labels
    weights: (B, C, D, H, W) - optional weight map
    """
    # Clamp for numerical stability and prevent log(0).
    eps = 1e-7
    input = torch.clamp(input, eps, 1.0 - eps)
    
    # Compute cross entropy: - target * log(input).
    loss = - target * torch.log(input)
    
    if weights is not None:
        loss = loss * weights
        
    # Sum over channels, then average over the batch and spatial voxels.
    return torch.mean(torch.sum(loss, dim=1))

def dice_loss_org_weights(pred, target, weights):
    """
    Revised standard Dice Loss with weights.
    Apply weights to both numerator and denominator, supporting boundary_importance > 1.
    """
    smooth = 1e-5

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)

    # Numerator: weighted.
    intersection = 2. * torch.sum(iflat * tflat * weights_flat)

    # Denominator must also be weighted; otherwise weights > 1 can make the loss negative.
    A_sum = torch.sum(iflat * iflat * weights_flat)
    B_sum = torch.sum(tflat * tflat * weights_flat)

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_II_weights(pred, target, weights):
    """
    Revised Dice Loss II with weights, optimized for foreground.
    Apply weights to both numerator and denominator.
    """
    smooth = 1e-5
    delta = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)

    # Smooth the prediction values using iflat / (iflat + delta).
    pred_smooth = iflat / (iflat + delta)

    # Numerator: weighted.
    intersection = 2. * torch.sum(pred_smooth * tflat * weights_flat)

    # Denominator must also be weighted; this part must remain enabled.
    A_sum = torch.sum(pred_smooth * pred_smooth * weights_flat)
    B_sum = torch.sum(tflat * tflat * weights_flat)

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

# -----------------------------------------------------------
# Unweighted versions and other helper functions; keep them unchanged.
# -----------------------------------------------------------

def dice_loss_org(pred, target):
    smooth = 1e-5
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = 2. * torch.sum(iflat * tflat)
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_II(pred, target):
    smooth = 1e-5
    delta = 0.1
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    pred_smooth = iflat / (iflat + delta)
    intersection = 2. * torch.sum(pred_smooth * tflat)
    A_sum = torch.sum(pred_smooth * pred_smooth)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_accuracy(pred, target):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = 2. * torch.sum(iflat * tflat)
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return intersection / (A_sum + B_sum + 1e-5)
    
