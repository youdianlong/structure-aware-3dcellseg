import numpy as np
import os
import torch
import time
import h5py
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.metrics import adapted_rand_error

# --- Import required project functions ---
from func.run_pipeline_super_vox import (
    semantic_segment_crop_and_cat_3_channel_output,
    img_3d_erosion_or_expansion, generate_super_vox_by_watershed,
    Cluster_Super_Vox, assign_boudary_voxels_to_cells_with_watershed,
    delete_too_small_cluster
)
# Import the network
from func.network_1219 import CellSegNet_1219
# Import utility functions
from func.utils import load_obj
# Import evaluation functions
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI

# ==========================================
#        Configuration / Hyperparameters
# ==========================================
# 1. Hardware and model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Path to the trained model; make sure it points to the correct file.
load_path = 'output/LRP/model_1219_LRP_best_seg.pkl'

# 2. Dataset path
LRP_dataset_root = 'D:\Dataset\LateralRootPrimordia_processed_wide_boundary'
# If using file names from dataset_info, make sure the path can be joined correctly.
dataset_info_path = "dataset_info/LRP_dataset_info"

# 3. Inference parameters
crop_cube_size = 128
stride = 64

# 4. Post-processing parameters (TASCAN)
# LRP cells are usually smaller and denser than Ovules cells, so these parameters may need fine-tuning.
how_close_are_the_super_vox_to_boundary = 2
min_touching_percentage = 0.40  # Slightly lower to avoid over-merging
min_cell_size_threshold = 10
min_touching_area = 5
scale_factor = 0.4  # Downsampling factor for faster IoU calculation


# --- 1. Initialize the model ---
print(f"Loading model from {load_path}...")
model = CellSegNet_1219(input_channel=1, n_classes=3)

if os.path.exists(load_path):
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    raise FileNotFoundError(f"Model file not found: {load_path}")

model.eval()
model.to(device)

# --- 2. Load LRP data (test set) ---
print("Loading Dataset Info...")
# Read dataset_info
LRP_info = load_obj(dataset_info_path)

# Get the test list, supporting either dictionary or list structures.
if 'test' in LRP_info:
    test_data = LRP_info['test']
    if isinstance(test_data, dict):
        test_list = list(test_data.keys())
    else:
        test_list = test_data
else:
    raise ValueError("Dataset info missing 'test' split")

# === Select the test case here ===
# Use the first case by default, or manually specify a file name such as case_name = "Movie1_t00010_crop_gt.h5".
#case_name = test_list[0]
case_name = 'Movie2_T00010_crop_gt.h5'
# Build the full path; LRP data is usually in the test subfolder.
case_path = os.path.join(LRP_dataset_root, 'test', case_name)
if not os.path.exists(case_path):
    # If it is not in the subfolder, try the root directory.
    case_path = os.path.join(LRP_dataset_root, case_name)

print(f"\nTesting case: {case_name}")
print(f"Path: {case_path}")

# Read H5 data (raw + label)
with h5py.File(case_path, 'r') as f:
    if 'raw' not in f:
        raise KeyError("H5 missing 'raw' data")

    # Support different ground-truth key names.
    if 'ins' in f:
        label_key = 'ins'
    elif 'label' in f:
        label_key = 'label'
    elif 'label_ins' in f:
        label_key = 'label_ins'
    else:
        raise KeyError("H5 missing 'ins' or 'label' data")

    raw_img = np.array(f['raw'][:], dtype=float)
    hand_seg = np.array(f[label_key][:], dtype=float)

print(f"Data Shape: {raw_img.shape}")

# --- 3. Model inference (Inference with TTA) ---
# Follow the Ovules script format and add three-view TTA.
print('Running Inference (TTA Enabled)...')
start_time = time.time()

raw_img_size = raw_img.shape
seg_foreground_comp = np.zeros(raw_img_size)
seg_boundary_comp = np.zeros(raw_img_size)


transposes = [[0, 1, 2]]
reverse_transposes = [[0, 1, 2]]
# # Define transforms for three directions.
# transposes = [
#     [0, 1, 2],
#     [2, 0, 1],
#     [1, 2, 0]
# ]
# # Define the corresponding inverse transforms.
# reverse_transposes = [
#     [0, 1, 2],
#     [1, 2, 0],
#     [2, 0, 1]
# ]

for idx, transpose in enumerate(transposes):
    print(f"  - View {idx + 1}: Transpose {transpose}")

    with torch.no_grad():
        # Feed the transposed image.
        seg_img = semantic_segment_crop_and_cat_3_channel_output(
            raw_img.transpose(transpose), model, device,
            crop_cube_size=crop_cube_size, stride=stride
        )

    # Extract results using foreground and boundary logic.
    seg_foreground = (seg_img['foreground'] > seg_img['background']) & (seg_img['foreground'] > seg_img['boundary'])
    seg_boundary = (seg_img['boundary'] > seg_img['background']) & (seg_img['boundary'] > seg_img['foreground'])

    # Core step: restore the original orientation (inverse transpose).
    seg_foreground = np.array(seg_foreground, dtype=float).transpose(reverse_transposes[idx])
    seg_boundary = np.array(seg_boundary, dtype=float).transpose(reverse_transposes[idx])

    # Accumulate results.
    seg_foreground_comp += seg_foreground
    seg_boundary_comp += seg_boundary

# Combine results by voting.
seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=int)
seg_foreground_comp = np.array(seg_foreground_comp > 0, dtype=int)
seg_foreground_comp[seg_boundary_comp > 0] = 0
seg_background_comp = np.array(1 - seg_foreground_comp - seg_boundary_comp > 0, dtype=int)

# --- 4. Generate instances by post-processing ---
print("Running Post-processing (TASCAN)...")

# 1. Erosion
seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(
    1 - seg_foreground_comp,
    kernel_size=how_close_are_the_super_vox_to_boundary + 1,
    device=device
)
# 2. Generate supervoxels
seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(
    seg_foreground_erosion,
    connectivity=min_touching_area
)
# 3. Clustering
cluster_super_vox = Cluster_Super_Vox(
    min_touching_area=min_touching_area,
    min_touching_percentage=min_touching_percentage
)
cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

# 4. Delete cells that are too small
seg_foreground_single_cell_with_boundary_delete_too_small = delete_too_small_cluster(
    seg_foreground_single_cell_with_boundary,
    threshold=min_cell_size_threshold
)
# 5. Watershed-based boundary assignment
seg_final = assign_boudary_voxels_to_cells_with_watershed(
    seg_foreground_single_cell_with_boundary_delete_too_small,
    seg_boundary_comp,
    seg_background_comp,
    compactness=1
)

end_time = time.time()
print(f"Inference & Post-processing Time: {end_time - start_time:.2f}s")

# ==========================================
#      Numerical Evaluation
# ==========================================
print("\n=== Calculating Metrics ===")

gt_flat = hand_seg.astype(int).flatten()
pred_flat = seg_final.astype(int).flatten()

print("Calculating ARI / ARE ...")
ARI = adjusted_rand_score(gt_flat, pred_flat)
ARE = adapted_rand_error(gt_flat, pred_flat)
VOI_val = VOI(seg_final.astype(int), hand_seg.astype(int))

print(f"ARI: {ARI:.4f}")
print(f"ARE: {ARE}")
print(f"VOI Total: {sum(VOI_val):.4f} (Split: {VOI_val[0]:.4f}, Merge: {VOI_val[1]:.4f})")


# Calculate individual cell metrics.
def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode=mode)
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])
    return img_3d


org_shape = seg_final.shape
output_size = (int(org_shape[0] * scale_factor), int(org_shape[1] * scale_factor), int(org_shape[2] * scale_factor))
print(f"Scaling for IoU/Dice: {scale_factor}x")

hand_seg_resized = img_3d_interpolate(hand_seg, output_size=output_size, mode='nearest')
seg_final_resized = img_3d_interpolate(seg_final, output_size=output_size, mode='nearest')

accuracy = IOU_and_Dice_Accuracy(hand_seg_resized, seg_final_resized)
accuracy_record = accuracy.cal_accuracy_II()

# Core fix: add the metric calculations for thresholds above 0.5.
iou_mean = np.mean(accuracy_record[:, 1])
dice_mean = np.mean(accuracy_record[:, 2])

iou_70 = np.mean(accuracy_record[:, 1] > 0.7)
dice_70 = np.mean(accuracy_record[:, 2] > 0.7)

iou_50 = np.mean(accuracy_record[:, 1] > 0.5)
dice_50 = np.mean(accuracy_record[:, 2] > 0.5)


print("-" * 30)
print(f"Current Parameters:")
print(f"  how_close_are_the_super_vox_to_boundary: {how_close_are_the_super_vox_to_boundary}")
print(f"  min_touching_percentage: {min_touching_percentage}")
print(f"  min_cell_size_threshold: {min_cell_size_threshold}")
print(f"  min_touching_area: {min_touching_area}")
print(f"  scale_factor: {scale_factor}")
print("-" * 30)

print("-" * 30)
print(f"Result for: {case_name}")
print(f"Avg IoU:  {iou_mean:.4f}")
print(f"Avg Dice: {dice_mean:.4f}")
print("-" * 15)
print(f"Accuracy (IoU > 0.7): {iou_70 * 100:.2f}%")
print(f"Accuracy (Dice > 0.7): {dice_70 * 100:.2f}%")
print(f"Accuracy (IoU > 0.5): {iou_50 * 100:.2f}%")
print(f"Accuracy (Dice > 0.5): {dice_50 * 100:.2f}%")
print("-" * 30)
