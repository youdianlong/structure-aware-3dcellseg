import numpy as np
import os
import torch
import time
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.metrics import adapted_rand_error

# --- Import required project functions ---
from func.run_pipeline_super_vox import (
    semantic_segment_crop_and_cat_3_channel_output,
    img_3d_erosion_or_expansion, generate_super_vox_by_watershed,
    Cluster_Super_Vox, assign_boudary_voxels_to_cells_with_watershed,
    delete_too_small_cluster
)
# Import networks
from func.network125 import CellSegNet_basic_vox_lite
from func.network import CellSegNet_basic_lite
# Import utility functions
from func.utils import load_obj
# Import evaluation functions
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI

# --- 1. Initialize the model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = CellSegNet_basic_vox_lite(input_channel=1, n_classes=3, output_func="softmax")
model = CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func="softmax")

# load_path = 'output/model_voxDNNA_seg.pkl'
load_path = 'output/model_Ovules.pkl'

print(f"Loading model from {load_path}...")
checkpoint = torch.load(load_path)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# --- 2. Load Ovules data ---
Ovules_data_dict = load_obj("dataset_info/Ovules_dataset_info")
crop_cube_size = 128
stride = 64
min_touching_area = 30

# You can loop through all test samples here, or evaluate only one.
case = 'N_435_final_crop_ds2.npz'
print(f"\nTesting case: {case}")

hf = np.load(Ovules_data_dict['test'][case])
raw_img = np.array(hf["raw"], dtype=float)
hand_seg = np.array(hf["ins"], dtype=float)  # Ground Truth

# --- 3. Model inference ---
print('Running Inference...')
start_time = time.time()

raw_img_size = raw_img.shape
seg_foreground_comp = np.zeros(raw_img_size)
seg_boundary_comp = np.zeros(raw_img_size)

transposes = [[0, 1, 2]]
reverse_transposes = [[0, 1, 2]]

for idx, transpose in enumerate(transposes):
    with torch.no_grad():
        seg_img = semantic_segment_crop_and_cat_3_channel_output(
            raw_img.transpose(transpose), model, device,
            crop_cube_size=crop_cube_size, stride=stride
        )

    # Extract results and restore the original orientation.
    seg_foreground = (seg_img['foreground'] > seg_img['background']) & (seg_img['foreground'] > seg_img['boundary'])
    seg_boundary = (seg_img['boundary'] > seg_img['background']) & (seg_img['boundary'] > seg_img['foreground'])

    # Simple conversion logic to avoid heavy argmax computation.
    seg_foreground = np.array(seg_foreground, dtype=float).transpose(reverse_transposes[idx])
    seg_boundary = np.array(seg_boundary, dtype=float).transpose(reverse_transposes[idx])

    seg_foreground_comp += seg_foreground
    seg_boundary_comp += seg_boundary

# Combine results.
seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=int)
seg_foreground_comp = np.array(seg_foreground_comp > 0, dtype=int)
seg_foreground_comp[seg_boundary_comp > 0] = 0
seg_background_comp = np.array(1 - seg_foreground_comp - seg_boundary_comp > 0, dtype=int)

# --- 4. Generate instances by post-processing ---
print("Running Post-processing (TASCAN)...")
how_close_are_the_super_vox_to_boundary = 2
min_touching_percentage = 0.51

seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(
    1 - seg_foreground_comp,
    kernel_size=how_close_are_the_super_vox_to_boundary + 1,
    device=device
)
seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(
    seg_foreground_erosion,
    connectivity=min_touching_area
)

cluster_super_vox = Cluster_Super_Vox(
    min_touching_area=min_touching_area,
    min_touching_percentage=min_touching_percentage
)
cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

min_cell_size_threshold = 10
seg_foreground_single_cell_with_boundary_delete_too_small = delete_too_small_cluster(
    seg_foreground_single_cell_with_boundary,
    threshold=min_cell_size_threshold
)
seg_final = assign_boudary_voxels_to_cells_with_watershed(
    seg_foreground_single_cell_with_boundary_delete_too_small,
    seg_boundary_comp,
    seg_background_comp,
    compactness=1
)

end_time = time.time()
print(f"Inference & Post-processing Time: {end_time - start_time:.2f}s")

# ==========================================
#      Numerical Evaluation Only
# ==========================================
print("\n=== Calculating Metrics ===")

# 1. Prepare data
gt_flat = hand_seg.astype(int).flatten()
pred_flat = seg_final.astype(int).flatten()

# 2. Calculate global clustering metrics (ARI, ARE, VOI)
ARI = adjusted_rand_score(gt_flat, pred_flat)
ARE = adapted_rand_error(gt_flat, pred_flat)
VOI_val = VOI(seg_final.astype(int), hand_seg.astype(int))

print(f"ARI: {ARI:.4f}")
print(f"ARE: {ARE}")
print(f"VOI Total: {sum(VOI_val):.4f} (Split: {VOI_val[0]:.4f}, Merge: {VOI_val[1]:.4f})")


# 3. Calculate individual cell metrics (IoU & Dice)
# Helper function: 3D interpolation and scaling
def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode=mode)
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])
    return img_3d


# Downsample to speed up computation (0.5 or 0.3)
scale_factor = 0.5
org_shape = seg_final.shape
output_size = (int(org_shape[0] * scale_factor), int(org_shape[1] * scale_factor), int(org_shape[2] * scale_factor))
print(f"Scaling for IoU/Dice calculation: {scale_factor}x")

hand_seg_resized = img_3d_interpolate(hand_seg, output_size=output_size, mode='nearest')
seg_final_resized = img_3d_interpolate(seg_final, output_size=output_size, mode='nearest')

accuracy = IOU_and_Dice_Accuracy(hand_seg_resized, seg_final_resized)
accuracy_record = accuracy.cal_accuracy_II()  # [id, iou, dice]

# 4. Print final results
iou_mean = np.mean(accuracy_record[:, 1])
dice_mean = np.mean(accuracy_record[:, 2])
iou_50 = np.mean(accuracy_record[:, 1] > 0.5)
dice_70 = np.mean(accuracy_record[:, 2] > 0.7)

print("-" * 30)
print(f"Avg IoU:  {iou_mean:.4f}")
print(f"Avg Dice: {dice_mean:.4f}")
print(f"Accuracy (IoU > 0.5): {iou_50 * 100:.2f}%")
print(f"Accuracy (Dice > 0.7): {dice_70 * 100:.2f}%")
print("-" * 30)
