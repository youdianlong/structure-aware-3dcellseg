# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import time

# --- 1. Import project dependencies ---
from func.load_dataset_domain import DomainGENDataset
from func.network125 import CellSegNet_basic_vox_lite
from func.loss_func1 import dice_accuracy, dice_loss_II, dice_loss_II_weights, dice_loss_org_weights, softmax_cross_entropy_loss
from func.voxel_adv import GRL, voxel_DomainDiscriminator

# --- 2. Hyperparameters ---
save_path = "output/model_ablation"
os.makedirs(save_path, exist_ok=True)

max_epoch = 400
learning_rate = 1e-4
batch_size = 8
train_img_crop_size = (64, 64, 64)
train_file_format1 = '.h5'  # ATAS
train_file_format2 = '.npz' # Ovules
# model_save_freq removed
boundary_importance = 4
num_workers = 8
domain_weight = 0.2

atas_dataset_path = '/data1/myt/dataset/ATAS_processed/'
ovules_dataset_path = '/data1/myt/dataset/ovules_processed_thin_boundary/'

# Helper function: load .pkl
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# --- 3. Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# --- 4. Data Preparation (Cycle/Oversampling Logic) ---
print("Loading datasets...")

# 4.1 Load ATAS (Source Domain)
atas_info = load_obj("dataset_info/ATAS_dataset_info")
# Convert to dict
atas_dict = {}
for i, name in enumerate(atas_info['train']):
    atas_dict['atas_{}'.format(i)] = os.path.join(atas_dataset_path, name)

# 4.2 Load Ovules (Target Domain) - Merge train, val, test
ovules_info = load_obj("dataset_info/Ovules_dataset_info")
ovules_paths = []

for split in ['train', 'val', 'test']:
    if split in ovules_info:
        for f in ovules_info[split]:
            ovules_paths.append(os.path.join(ovules_dataset_path, split, f))

# Fallback if splits are missing
if not ovules_paths:
    print("Warning: Ovules splits not found, using 'train' key only.")
    for f in ovules_info.get('train', []):
        ovules_paths.append(os.path.join(ovules_dataset_path, 'train', f))

# 4.3 [Key Modification] Cycle / Oversampling Logic
num_atas = len(atas_dict)
num_ovules = len(ovules_paths)

print("Original Data -> ATAS: {}, Ovules: {}".format(num_atas, num_ovules))

if num_ovules > 0 and num_atas > num_ovules:
    # Calculate repetition factor
    repeat_times = num_atas // num_ovules
    if num_atas % num_ovules != 0:
        repeat_times += 1
    
    print("Executing Cycle (Oversampling) on Ovules... Repeating {} times".format(repeat_times))
    ovules_paths = ovules_paths * repeat_times

# Generate Ovules dict with unique keys
ovules_dict = {}
for i, p in enumerate(ovules_paths):
    ovules_dict['ovules_{}'.format(i)] = p

print("Final Data (After Cycle) -> ATAS: {}, Ovules: {}".format(len(atas_dict), len(ovules_dict)))

# Initialize Dataset
dataset = DomainGENDataset(data_dict1=atas_dict, data_dict2=ovules_dict)

dataset.set_para(
    file_format1=train_file_format1,
    file_format2=train_file_format2,
    crop_size=train_img_crop_size,
    boundary_importance=boundary_importance,
    need_tensor_output=True,
    need_transform=True
)

# Initialize DataLoader
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,  # Must be True to mix domains
    num_workers=num_workers,
    pin_memory=True
)
print("DataLoader ready.")

# --- 5. Initialize Models ---
model_GC = CellSegNet_basic_vox_lite(input_channel=1, n_classes=3).to(device)

# Initialize 4 Discriminators (Multi-scale)
# Assuming features are [32, 64, 64, 64]
D1 = voxel_DomainDiscriminator(in_features=32, num_domains=1).to(device)
D2 = voxel_DomainDiscriminator(in_features=64, num_domains=1).to(device)
D3 = voxel_DomainDiscriminator(in_features=64, num_domains=1).to(device)
D4 = voxel_DomainDiscriminator(in_features=64, num_domains=1).to(device)

discriminator_list = [D1, D2, D3, D4]
grl = GRL(alpha=1.0) 

# --- 6. Optimizer ---
optimizer = optim.Adam(
    list(model_GC.parameters()) +
    list(D1.parameters()) +
    list(D2.parameters()) +
    list(D3.parameters()) +
    list(D4.parameters()),
    lr=learning_rate
)

# --- 7. Loss Functions ---
loss_func_domain = nn.BCEWithLogitsLoss()

# --- 8. Training Loop ---
print("--- Start Training ---")
minimum = float('inf') # Initialize to infinity

for epoch in range(max_epoch):
    model_GC.train()
    for d in discriminator_list:
        d.train()

    running_seg_loss = 0.0
    running_domain_loss = 0.0
    batch_count = 0
    
    for i, batch in enumerate(train_loader):
        # 8.1. Prepare Data
        images = batch['raw'].to(device)
        domain_labels = batch['domain_label'].to(device).long()

        # Prepare Segmentation Ground Truth
        seg_groundtruth_f = torch.tensor(batch['foreground'] > 0, dtype=torch.float).to(device)
        seg_groundtruth_bb = torch.cat((
            torch.tensor(batch['background'] > 0, dtype=torch.float),
            torch.tensor(batch['boundary'] > 0, dtype=torch.float)
        ), dim=1).to(device)

        weights_f = batch['weights_foreground'].to(device)
        weights_bb = torch.cat((batch['weights_background'], batch['weights_boundary']), dim=1).to(device)

        # 8.2. Forward
        optimizer.zero_grad()

        # (G+C) Branch: Get segmentation output AND feature list
        seg_output, feature_list = model_GC(images, return_features=True)

        # (D) Branch: Multi-scale Adversarial
        total_domain_loss = 0.0
        layer_weights = [0.2, 0.3, 0.8, 1.0]
        target_labels_raw = batch['domain_label'].to(device).float()

        for idx, (feat, D_net) in enumerate(zip(feature_list, discriminator_list)):
            # GRL
            feat_reversed = grl(feat)
            # D Forward
            domain_logits = D_net(feat_reversed)
            
            # Expand labels to match output size
            B_curr, _, D_curr, H_curr, W_curr = domain_logits.shape
            target_maps = target_labels_raw.view(B_curr, 1, 1, 1, 1).expand(B_curr, 1, D_curr, H_curr, W_curr)
            
            # BCE Loss
            loss_curr = loss_func_domain(domain_logits, target_maps)
            total_domain_loss += loss_curr * layer_weights[idx]

        # --- Calculate Segmentation Loss ---
        seg_output_f = seg_output[:, 2, :, :, :]
        seg_output_bb = torch.cat((seg_output[:, 0, :, :, :], seg_output[:, 1, :, :, :]), dim=1)

        loss_dice = dice_loss_org_weights(seg_output_bb, seg_groundtruth_bb, weights_bb) + \
                    dice_loss_II_weights(seg_output_f, seg_groundtruth_f, weights_f)
        
        seg_groundtruth_all = torch.cat((
            torch.tensor(batch['background'] > 0, dtype=torch.float).to(device),
            torch.tensor(batch['boundary'] > 0, dtype=torch.float).to(device),
            torch.tensor(batch['foreground'] > 0, dtype=torch.float).to(device)), dim=1)
        
        weights_all = torch.cat((batch['weights_background'], batch['weights_boundary'], batch['weights_foreground']), dim=1).to(device)
        
        loss_ce = softmax_cross_entropy_loss(seg_output, seg_groundtruth_all, weights_all)
        loss_seg = loss_dice + loss_ce

        # Domain loss assignment
        loss_domain = total_domain_loss

        # 8.4. Backward
        total_loss = loss_seg + loss_domain * domain_weight

        total_loss.backward()
        optimizer.step()

        running_seg_loss += loss_seg.item()
        running_domain_loss += loss_domain.item()
        batch_count += 1

        # 8.5. Logging
        if i % 10 == 0:
            accuracy = dice_accuracy(seg_output_f.detach(), seg_groundtruth_f.detach())
            print("Epoch [{}/{}], Step [{}/{}], Total: {:.4f}, Seg: {:.4f}, Acc: {:.4f}, Dom: {:.4f}".format(
                epoch + 1, max_epoch, i, len(train_loader), 
                total_loss.item(), loss_seg.item(), accuracy.item(), loss_domain.item()))

    # --- 9. Save Logic ---
    avg_seg_loss = running_seg_loss / batch_count
    
    # Save Best Seg Loss ONLY
    if avg_seg_loss < minimum:
        minimum = avg_seg_loss
        save_model_path = os.path.join(save_path, "cycleOvules_advonly.pkl")
        torch.save(model_GC.state_dict(), save_model_path)
        print("Update Best Seg: {:.5f}".format(minimum))

print("Final Best Seg: {}".format(minimum))
print("--- Training Finished ---")