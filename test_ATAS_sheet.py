import numpy as np
import os
import pickle
import copy
import edt
import time
import cv2
import pandas as pd
import h5py

# Import evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.metrics import adapted_rand_error

# Import PyTorch
import torch
from torch import from_numpy as from_numpy

# Import custom functions and keep the original references
from func.run_pipeline_super_vox import segment_super_vox_3_channel, semantic_segment_crop_and_cat_3_channel_output, \
    img_3d_erosion_or_expansion, \
    generate_super_vox_by_watershed, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, Cluster_Super_Vox, \
    assign_boudary_voxels_to_cells_with_watershed, \
    delete_too_small_cluster, reassign
from func.run_pipeline import segment, assign_boudary_voxels_to_cells, dbscan_of_seg, semantic_segment_crop_and_cat
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI
from func.network import VoxResNet, CellSegNet_basic_lite
from func.network125 import CellSegNet_basic_vox_lite
from func.network_1219 import CellSegNet_1219
from func.unet_3d_basic import UNet3D_basic
from func.utils import save_obj, load_obj

# ================= 1. Model Initialization =================
# model = CellSegNet_basic_vox_lite(input_channel=1, n_classes=3, output_func="softmax")
# model=CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func = "softmax")
# mode = UNet3D_basic(in_channels=1,out_channels=3)
# model = VoxResNet(input_channel=1,n_classes=3,output_func='softmax')
model = CellSegNet_1219(input_channel=1,n_classes=3)

#load_path = 'output/model_voxDNNA_seg.pkl'
#load_path = 'output/model_Base_unet3d_best.pkl'
#load_path = 'output/model_Base_VoxResNet_best.pkl'
load_path = 'output/ATAS/model_1219_best_seg.pkl'
print(f"Loading model from: {load_path}")

checkpoint = torch.load(load_path)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ================= 2. Data Loading =================
ATAS_data_dict = load_obj("dataset_info/ATAS_dataset_info")

crop_cube_size = 64
stride = 64
min_touching_area = 2

print("Available test cases: " + str(ATAS_data_dict['test'].keys()))
case = 'plant15_0hrs.h5'
print(f"Processing test case: {case}")
base_dir = "D:/Dataset/ATAS_processed/"

# 3. Core step: join the base path with the case file name.
# This is the intended behavior: append the specific file name to the base path.
# Equivalent logic: full_path = base_dir + case
full_path = os.path.join(base_dir, case)

print(f"正在读取文件: {full_path}")

hf = h5py.File(full_path, 'r')

raw_img = np.array(hf["raw"], dtype=float)
hand_seg = np.array(hf["ins"], dtype=float)
boundary_gt = np.array(hf["boundary"], dtype=float)
# background_gt and foreground_gt are not used in evaluation here, but keep the loading logic as a safeguard.
background_gt = np.array(hf["background"], dtype=float)
foreground_gt = np.array(hf["foreground"], dtype=float)

print(f"Raw img shape: {raw_img.shape}")
print(f"Hand seg shape: {hand_seg.shape}")

start = time.time()

# ================= 3. Model Inference =================
raw_img_size = raw_img.shape
seg_background_comp = np.zeros(raw_img_size)
seg_boundary_comp = np.zeros(raw_img_size)

transposes = [[0, 1, 2]]  # Add [2,0,1] or [0,2,1] here if TTA is needed.
reverse_transposes = [[0, 1, 2]]
#transposes = [[0, 1, 2], [0, 2, 1]]  # Add [2,0,1] or [0,2,1] here if TTA is needed.
#reverse_transposes = [[0, 1, 2], [0, 2, 1]]

for idx, transpose in enumerate(transposes):
    print(f"{idx + 1}: Transpose {transpose} processing...")
    with torch.no_grad():
        seg_img = semantic_segment_crop_and_cat_3_channel_output(
            raw_img.transpose(transpose),
            model,
            device,
            crop_cube_size=crop_cube_size,
            stride=stride
        )

    seg_img_background = seg_img['background']
    seg_img_boundary = seg_img['boundary']
    seg_img_foreground = seg_img['foreground']
    torch.cuda.empty_cache()

    # argmax
    print('Argmax calculation...', end='\r')
    seg = []
    seg.append(seg_img_background)
    seg.append(seg_img_boundary)
    seg.append(seg_img_foreground)
    seg = np.array(seg)
    seg_argmax = np.argmax(seg, axis=0)

    # Convert probability map to 0/1 segment
    seg_background = np.zeros(seg_img_background.shape)
    seg_background[np.where(seg_argmax == 0)] = 1
    seg_foreground = np.zeros(seg_img_foreground.shape)
    seg_foreground[np.where(seg_argmax == 2)] = 1
    seg_boundary = np.zeros(seg_img_boundary.shape)
    seg_boundary[np.where(seg_argmax == 1)] = 1

    # Revert transpose
    seg_background = seg_background.transpose(reverse_transposes[idx])
    seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
    seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

    seg_background_comp += seg_background
    seg_boundary_comp += seg_boundary

print("\nModel semantic segmentation completed.")
seg_background_comp = np.array(seg_background_comp > 0, dtype=int)
seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=int)
seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp > 0, dtype=int)

end = time.time()
print(f"Inference time elapsed: {end - start:.2f} s")

# ================= 4. Post-processing (WaterShed & Clustering) =================
print("Starting post-processing...")

how_close_are_the_super_vox_to_boundary = 3
min_touching_percentage = 0.60
min_cell_size_threshold = 50


seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(
    1 - seg_foreground_comp,
    kernel_size=how_close_are_the_super_vox_to_boundary + 1,
    device=device
)
seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion)
print(f"There are {len(np.unique(seg_foreground_super_voxel_by_ws))} super voxels.")

# Super voxel clustering
cluster_super_vox = Cluster_Super_Vox(
    min_touching_area=min_touching_area,
    min_touching_percentage=min_touching_percentage
)
cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

# Delete too small cells

seg_foreground_single_cell_with_boundary_delete_too_small = delete_too_small_cluster(
    seg_foreground_single_cell_with_boundary,
    threshold=min_cell_size_threshold
)

# Assign boundary voxels to cells
seg_final = assign_boudary_voxels_to_cells_with_watershed(
    seg_foreground_single_cell_with_boundary_delete_too_small,
    seg_boundary_comp,
    seg_background_comp,
    compactness=1
)
print("Post-processing completed.")

# ================= 5. Numerical Evaluation =================
print("\n=== Calculating Metrics ===")

# Metric Set 1: ARI, ARE, VOI
print("Calculating ARI, ARE, VOI...")
ARE = adapted_rand_error(hand_seg.astype(int).flatten(), seg_final.astype(int).flatten())
ARI = adjusted_rand_score(hand_seg.flatten(), seg_final.flatten())
VOI_val = VOI(seg_final.astype(int), hand_seg.astype(int))

print(f"ARI: {ARI}")
print(f"ARE: {ARE}")
print(f"VOI: {VOI_val}")


# Metric Set 2: IoU & Dice with downsampling optimization
def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode=mode)
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])
    return img_3d


scale_factor = 0.3
org_shape = seg_final.shape
output_size = (int(org_shape[0] * scale_factor), int(org_shape[1] * scale_factor), int(org_shape[2] * scale_factor))
print(f"Downsampling for accuracy calculation: {org_shape} --> {output_size}")

accuracy = IOU_and_Dice_Accuracy(
    img_3d_interpolate(hand_seg, output_size=output_size),
    img_3d_interpolate(seg_final, output_size=output_size)
)
accuracy_record = accuracy.cal_accuracy_II()

# Report Results
iou_70 = np.mean(accuracy_record[:, 1] > 0.7)
dice_70 = np.mean(accuracy_record[:, 2] > 0.7)
iou_50 = np.mean(accuracy_record[:, 1] > 0.5)
dice_50 = np.mean(accuracy_record[:, 2] > 0.5)

print('-' * 40)
print(f'Cell Count Accuracy (IoU > 0.7):  {iou_70:.4f}')
print(f'Cell Count Accuracy (Dice > 0.7): {dice_70:.4f}')
print(f'Cell Count Accuracy (IoU > 0.5):  {iou_50:.4f}')
print(f'Cell Count Accuracy (Dice > 0.5): {dice_50:.4f}')
print(f'Average IoU:  {np.mean(accuracy_record[:, 1]):.4f}')
print(f'Average Dice: {np.mean(accuracy_record[:, 2]):.4f}')
print('-' * 40)
