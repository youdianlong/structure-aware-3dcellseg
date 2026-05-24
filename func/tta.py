# func/tta.py
import torch
import torch.nn as nn
import numpy as np


class TTAModel(nn.Module):
    """
    Test-time augmentation (TTA) wrapper.
    Wrap the original model to automatically perform multi-view inference and result fusion during forward.

    Args:
        model (nn.Module): trained base segmentation model
        device (torch.device): runtime device
    """

    def __init__(self, model, device):
        super(TTAModel, self).__init__()
        self.model = model
        self.device = device
        self.model.eval()  # Force evaluation mode.

        # Define augmentation view configurations.
        # List element: {'flip': flip axis (2/3/4), 'perm': forward axis permutation, 'inv_perm': inverse permutation}.
        # Note: tensor dimensions are (B, C, D, H, W), corresponding to indices (0, 1, 2, 3, 4).
        self.transforms = [
            # 1. Original view (identity)
            {'flip': None, 'perm': (0, 1, 2, 3, 4), 'inv_perm': (0, 1, 2, 3, 4)},

            # 2. Axis flips (flip D, H, W)
            {'flip': [2], 'perm': (0, 1, 2, 3, 4), 'inv_perm': (0, 1, 2, 3, 4)},
            {'flip': [3], 'perm': (0, 1, 2, 3, 4), 'inv_perm': (0, 1, 2, 3, 4)},
            {'flip': [4], 'perm': (0, 1, 2, 3, 4), 'inv_perm': (0, 1, 2, 3, 4)},

            # 3. Axis permutations, assuming the input patch is an isotropic cube such as 64x64x64.
            # (D, H, W) -> (H, W, D)
            {'flip': None, 'perm': (0, 1, 3, 4, 2), 'inv_perm': (0, 1, 4, 2, 3)},
            # (D, H, W) -> (W, D, H)
            {'flip': None, 'perm': (0, 1, 4, 2, 3), 'inv_perm': (0, 1, 3, 4, 2)},
        ]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input image (B, 1, D, H, W)
        Returns:
            avg_prob (torch.Tensor): fused probability map (B, n_classes, D, H, W)
        """
        pred_sum = None

        # Iterate over all views.
        for t in self.transforms:
            # --- A. Transform input (augment) ---
            aug_x = x
            # 1. Axis permutation
            if t['perm'] != (0, 1, 2, 3, 4):
                aug_x = aug_x.permute(t['perm'])
            # 2. Flip
            if t['flip']:
                aug_x = torch.flip(aug_x, dims=t['flip'])

            # --- B. Model inference ---
            # Ensure gradients are not computed.
            with torch.no_grad():
                # The original model may return (prob, feature) or prob, so handle both cases.
                output = self.model(aug_x)
                if isinstance(output, tuple):
                    prob_map = output[0]  # If a tuple is returned, use the first element (probability map).
                else:
                    prob_map = output

            # --- C. Inverse-transform output (de-augment) ---
            # 1. Inverse flip
            if t['flip']:
                prob_map = torch.flip(prob_map, dims=t['flip'])
            # 2. Inverse axis permutation
            if t['inv_perm'] != (0, 1, 2, 3, 4):
                prob_map = prob_map.permute(t['inv_perm'])

            # --- D. Accumulate probabilities ---
            if pred_sum is None:
                pred_sum = prob_map
            else:
                pred_sum += prob_map

            # Clear GPU memory promptly.
            # del aug_x, prob_map
            # torch.cuda.empty_cache()

        # Average by soft voting.
        avg_prob = pred_sum / len(self.transforms)

        return avg_prob
