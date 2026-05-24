# add SE block and ASPP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import time

# --- 1. Import modules ---
# Make sure func/load_dataset_domain.py exists.
from func.load_dataset_domain import DomainGENDataset
# Core upgrade: use the new SE+ASPP network.
from func.network_1219 import CellSegNet_1219
from func.loss_func1 import dice_accuracy, dice_loss_II_weights, dice_loss_org_weights
# Make sure func/voxel_adv.py exists.
from func.voxel_adv import GRL, voxel_DomainDiscriminator


# Define CE Loss.
def softmax_cross_entropy_loss(input, target, weights=None):
    eps = 1e-7
    input = torch.clamp(input, eps, 1.0 - eps)
    loss = - target * torch.log(input)
    if weights is not None:
        loss = loss * weights
    return torch.mean(torch.sum(loss, dim=1))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# --- 2. Hyperparameter settings ---
save_path = "output/model_1219_Supervised_DANN"  # Recommended to use a new directory.
max_epoch = 800
learning_rate = 1e-4
batch_size = 8
train_img_crop_size = (64, 64, 64)
train_file_format1 = '.h5'  # ATAS
train_file_format2 = '.npz'  # Ovules
model_save_freq = 80
boundary_importance = 4
num_workers = 8
domain_weight = 0.2  # Adversarial loss weight

# Dataset paths
atas_dataset_path = '/data1/myt/dataset/ATAS_processed/'
ovules_dataset_path = '/data1/myt/dataset/ovules_processed_thin_boundary/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_path, exist_ok=True)
print(f"Using device: {device}")

# --- 3. Data loading ---
print("正在加载数据集...")
atas_info = load_obj("dataset_info/ATAS_dataset_info")
atas_dict = {f'atas_{i}': os.path.join(atas_dataset_path, name) for i, name in enumerate(atas_info['train'])}

ovules_info = load_obj("dataset_info/Ovules_dataset_info")
# Process Ovules paths by merging Train/Val/Test.
ovules_paths = []
for split in ['train', 'val', 'test']:
    if split in ovules_info:
        ovules_paths.extend([os.path.join(ovules_dataset_path, split, f) for f in ovules_info[split]])
# If info directly stores file names and all files are under train, keep the original style; this is a general fallback.
if not ovules_paths:  # fallback
    ovules_paths = [os.path.join(ovules_dataset_path, 'train', f) for f in ovules_info['train']]

ovules_dict = {f'ovules_{i}': p for i, p in enumerate(ovules_paths)}

# Instantiate the Dataset.
dataset = DomainGENDataset(data_dict1=atas_dict, data_dict2=ovules_dict)
dataset.set_para(
    file_format1=train_file_format1,
    file_format2=train_file_format2,
    crop_size=train_img_crop_size,
    boundary_importance=boundary_importance,
    need_tensor_output=True,
    need_transform=True
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # Must be True to mix domains
    num_workers=num_workers,
    pin_memory=True
)
print(f"Dataset Ready. ATAS: {len(atas_dict)}, Ovules: {len(ovules_dict)}")

# --- 4. Model initialization ---
# Core upgrade: use Network_1219 (SE + ASPP).
model_GC = CellSegNet_1219(input_channel=1, n_classes=3).to(device)

# Key fix: discriminator channel counts must match the output features of Network_1219.
# Network_1219 feature channels are [32, 64, 128, 256], unlike the old model's [32, 64, 64, 64].
D1 = voxel_DomainDiscriminator(in_features=32, num_domains=1).to(device)
D2 = voxel_DomainDiscriminator(in_features=64, num_domains=1).to(device)
D3 = voxel_DomainDiscriminator(in_features=128, num_domains=1).to(device)
D4 = voxel_DomainDiscriminator(in_features=256, num_domains=1).to(device)

discriminator_list = [D1, D2, D3, D4]
grl = GRL(alpha=1.0)
loss_func_domain = nn.BCEWithLogitsLoss()

# The optimizer includes all parameters.
optimizer = optim.Adam(
    list(model_GC.parameters()) +
    list(D1.parameters()) + list(D2.parameters()) +
    list(D3.parameters()) + list(D4.parameters()),
    lr=learning_rate
)

# --- 5. Training loop ---
print("--- 开始训练 (Full Supervised + Domain Adversarial) ---")
min_seg_loss = float('inf')
best_composite_metric = float('-inf')

for epoch in range(max_epoch):
    model_GC.train()
    for d in discriminator_list: d.train()

    running_seg_loss = 0.0
    running_domain_loss = 0.0
    batch_count = 0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        images = batch['raw'].to(device)
        domain_labels = batch['domain_label'].to(device)  # 0=ATAS, 1=Ovules

        # -----------------------------------------------------------
        # 1. Prepare ground truth for the full batch without splitting by domain.
        # -----------------------------------------------------------
        # Ovules also has labels, so the data in the batch can be used directly.
        seg_groundtruth_f = torch.tensor(batch['foreground'] > 0, dtype=torch.float).to(device)
        # Background + boundary
        seg_groundtruth_bb = torch.cat((
            torch.tensor(batch['background'] > 0, dtype=torch.float),
            torch.tensor(batch['boundary'] > 0, dtype=torch.float)
        ), dim=1).to(device)

        # Weights
        weights_f = batch['weights_foreground'].to(device)
        weights_bb = torch.cat((batch['weights_background'], batch['weights_boundary']), dim=1).to(device)

        # -----------------------------------------------------------
        # 2. Forward propagation
        # -----------------------------------------------------------
        optimizer.zero_grad()

        # Network_1219 returns: seg_output, [c1, c2, c3, bn]
        seg_output, feature_list = model_GC(images, return_features=True)

        # -----------------------------------------------------------
        # 3. Compute segmentation loss (fully supervised).
        # -----------------------------------------------------------
        # Split predictions.
        seg_output_f = seg_output[:, 2, :, :, :]
        seg_output_bb = torch.cat((seg_output[:, 0, :, :, :], seg_output[:, 1, :, :, :]), dim=1)

        # Dice Loss
        loss_dice = dice_loss_org_weights(seg_output_bb, seg_groundtruth_bb, weights_bb) + \
                    dice_loss_II_weights(seg_output_f, seg_groundtruth_f, weights_f)

        # CE Loss
        seg_groundtruth_all = torch.cat((
            torch.tensor(batch['background'] > 0, dtype=torch.float).to(device),
            torch.tensor(batch['boundary'] > 0, dtype=torch.float).to(device),
            torch.tensor(batch['foreground'] > 0, dtype=torch.float).to(device)
        ), dim=1)
        weights_all = torch.cat((batch['weights_background'], batch['weights_boundary'], batch['weights_foreground']),
                                dim=1).to(device)

        loss_ce = softmax_cross_entropy_loss(seg_output, seg_groundtruth_all, weights_all)

        # Total segmentation loss
        loss_seg = loss_dice + loss_ce

        # -----------------------------------------------------------
        # 4. Compute domain adversarial loss (DANN regularization).
        # -----------------------------------------------------------
        total_domain_loss = 0.0
        # Feature-level weights for the 1219 network: [c1, c2, c3, bn]
        layer_weights = [0.1, 0.2, 0.5, 1.0]
        target_labels_float = domain_labels.float()

        for idx, (feat, D_net) in enumerate(zip(feature_list, discriminator_list)):
            # GRL reverses the gradient.
            feat_reversed = grl(feat)

            # Discriminator prediction
            d_logits = D_net(feat_reversed)

            # Expand labels to match feature-map dimensions (pixel-wise loss).
            B, _, D, H, W = d_logits.shape
            target_map = target_labels_float.view(B, 1, 1, 1, 1).expand(B, 1, D, H, W)

            loss_d = loss_func_domain(d_logits, target_map)
            total_domain_loss += loss_d * layer_weights[idx]

        # -----------------------------------------------------------
        # 5. Backpropagation
        # -----------------------------------------------------------
        total_loss = loss_seg + total_domain_loss * domain_weight

        total_loss.backward()
        optimizer.step()

        running_seg_loss += loss_seg.item()
        running_domain_loss += total_domain_loss.item()
        batch_count += 1

        # Print partial logs.
        if i % 20 == 0:
            acc = dice_accuracy(seg_output_f.detach(), seg_groundtruth_f.detach())
            print(f"Ep [{epoch + 1}/{max_epoch}] Step [{i}] "
                  f"Seg: {loss_seg.item():.4f} Dom: {total_domain_loss.item():.4f} Acc: {acc:.4f}")

    # --- End-of-epoch statistics ---
    avg_seg_loss = running_seg_loss / max(batch_count, 1)
    avg_domain_loss = running_domain_loss / max(batch_count, 1)

    # Composite metric calculation logic.
    domain_display = abs(avg_domain_loss - 0.693)  # 0.693 is log(2), the loss when the discriminator is fully confused.
    # Adjust this formula based on the actual preference, for example when lower seg_loss is preferred.
    # Here we assume a lower Composite_Metric is better.
    composite_metric = avg_seg_loss + 0.1 * domain_display

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1} Done in {epoch_time:.0f}s. Avg Seg Loss: {avg_seg_loss:.4f}")

    # --- Saving strategy ---
    # 1. Save the model with the lowest segmentation loss, the most practical metric.
    if avg_seg_loss < min_seg_loss:
        min_seg_loss = avg_seg_loss
        torch.save(model_GC.state_dict(), os.path.join(save_path, "model_1219_best_seg.pkl"))
        print(">>> Saved Best Seg Model")

    # 2. Save periodically.
    if (epoch + 1) % model_save_freq == 0:
        save_file = os.path.join(save_path, f"model_1219_epoch_{epoch + 1}.pkl")
        torch.save(model_GC.state_dict(), save_file)
        print(f"Checkpointed: {save_file}")

print("--- 训练完成 ---")
