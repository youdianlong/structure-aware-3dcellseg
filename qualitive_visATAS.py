import numpy as np
import os
import torch
import time
import h5py
import cv2
import matplotlib

# Windows/server crash prevention settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe  # Text outline effects
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.metrics import adapted_rand_error

# --- Import required project functions ---
from func.run_pipeline_super_vox0103 import (
    semantic_segment_crop_and_cat_3_channel_output,
    img_3d_erosion_or_expansion, generate_super_vox_by_watershed,
    Cluster_Super_Vox, assign_boudary_voxels_to_cells_with_watershed,
    delete_too_small_cluster,
    absorb_small_fragments,  # Innovation 1
    fill_black_holes  # Innovation 2
)
from func.network_1219 import CellSegNet_1219
from func.utils import load_obj
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI

# ==========================================
#        1. Configuration / Parameters
# ==========================================
# Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
load_path = 'output/ATAS/model_1219_best_seg.pkl'
ATAS_dataset_root = 'D:/Dataset/ATAS_processed/'
dataset_info_path = "dataset_info/ATAS_dataset_info"
save_img_dir = "output/img_results/ATAS_1219/"

# Inference parameters
crop_cube_size = 128
stride = 64

# Post-processing parameters
how_close_are_the_super_vox_to_boundary = 3
min_touching_percentage = 0.60
min_cell_size_threshold = 50
absorb_size_threshold = 40
min_touching_area = 2

# Visualization and evaluation parameters
VISUALIZE_SLICE_IDX = 60  # Slice index
VISUALIZE_AXIS = 0  # 0=Z axis, 1=Y axis, 2=X axis
PIXEL_SIZE_UM = 0.25  # Pixel size
SCALE_FACTOR = 0.4  # Downsampling factor for faster evaluation

os.makedirs(save_img_dir, exist_ok=True)


# ==========================================
#        2. Visualization Functions
# ==========================================
def colorful_seg_2d(seg):
    """2D Label to RGB"""
    unique_vals, val_counts = np.unique(seg, return_counts=True)
    background_val = unique_vals[np.argsort(val_counts)[::-1][0]]

    mask_gray = cv2.normalize(src=seg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    seg_RGB = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

    for unique_val in unique_vals:
        if unique_val == background_val:
            COLOR = np.array([0, 0, 0], dtype=int)
        else:
            np.random.seed(int(unique_val))
            COLOR = np.array(np.random.choice(np.arange(50, 255), size=3, replace=False), dtype=int)
        locs = np.where(seg == unique_val)
        seg_RGB[locs[0], locs[1], :] = COLOR
    return seg_RGB


def visualize_3_panel(raw, gt_instance, pred_instance, save_name, pixel_size=0.25):
    """
    Generate a concise three-panel figure with a clear numeric scale bar.
    """
    # 1. Enhance the raw image
    p1, p99 = np.percentile(raw, (1, 99))
    raw_show = np.clip((raw - p1) / (p99 - p1), 0, 1)

    # 2. Colorize labels
    gt_color = colorful_seg_2d(gt_instance.astype(int))
    pred_color = colorful_seg_2d(pred_instance.astype(int))

    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Raw
    axes[0].imshow(raw_show, cmap='gray')
    axes[0].set_title("Raw Image", fontsize=16, fontweight='bold')

    # GT
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth", fontsize=16, fontweight='bold')

    # Pred
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction (Ours)", fontsize=16, fontweight='bold')

    # 4. Scale bar and numeric label
    h, w = raw.shape
    scale_bar_um = 5
    scale_bar_px = int(scale_bar_um / pixel_size)
    label_text = f"{scale_bar_um} µm"

    for ax in axes:
        ax.axis('off')
        # Position calculation
        bar_x = w - scale_bar_px - 15
        bar_y = h - 15

        # White bar
        rect = patches.Rectangle((bar_x, bar_y), scale_bar_px, 4, linewidth=0, facecolor='white')
        ax.add_patch(rect)

        # Text with black outline
        ax.text(bar_x + scale_bar_px / 2, bar_y - 5, label_text,
                color='white', ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Figure Saved: {save_name}")


def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, *img_3d.shape)
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode=mode)
    img_3d = img_3d.detach().cpu().numpy()
    return img_3d.reshape(output_size)


# ==========================================
#        3. Main Pipeline
# ==========================================
if __name__ == "__main__":
    # --- A. Model initialization ---
    print(f"Loading model from {load_path}...")
    model = CellSegNet_1219(input_channel=1, n_classes=3)
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    else:
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model.eval().to(device)

    # --- B. Load data ---
    print("Loading Dataset Info...")
    dataset_info = load_obj(dataset_info_path)
    if 'test' in dataset_info:
        test_data = dataset_info['test']
        test_list = list(test_data.keys()) if isinstance(test_data, dict) else test_data
    else:
        test_list = dataset_info if isinstance(dataset_info, list) else []

    # Case Selection
    case_name = 'plant15_0hrs.h5'
    case_path = os.path.join(ATAS_dataset_root, 'test', case_name)
    if not os.path.exists(case_path):
        case_path = os.path.join(ATAS_dataset_root, case_name)

    print(f"\nProcessing Case: {case_name}")
    with h5py.File(case_path, 'r') as f:
        raw_img = np.array(f['raw'][:], dtype=float)
        for key in ['ins', 'label', 'label_ins', 'segmentation']:
            if key in f:
                hand_seg = np.array(f[key][:], dtype=float)
                break
    print(f"Data Shape: {raw_img.shape}")

    # --- C. Inference (TTA) ---
    print('Running Inference (TTA)...')
    start_time = time.time()
    raw_img_size = raw_img.shape
    seg_foreground_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)
    transposes = [[0, 1, 2]]

    for idx, transpose in enumerate(transposes):
        with torch.no_grad():
            seg_img = semantic_segment_crop_and_cat_3_channel_output(
                raw_img.transpose(transpose), model, device,
                crop_cube_size=crop_cube_size, stride=stride
            )
        seg_fg = (seg_img['foreground'] > seg_img['background']) & (seg_img['foreground'] > seg_img['boundary'])
        seg_bd = (seg_img['boundary'] > seg_img['background']) & (seg_img['boundary'] > seg_img['foreground'])

        rev_tr = [0, 1, 2]
        seg_foreground_comp += seg_fg.astype(float)
        seg_boundary_comp += seg_bd.astype(float)

    seg_boundary_comp = (seg_boundary_comp > 0).astype(int)
    seg_foreground_comp = (seg_foreground_comp > 0).astype(int)
    seg_foreground_comp[seg_boundary_comp > 0] = 0
    seg_background_comp = (1 - seg_foreground_comp - seg_boundary_comp > 0).astype(int)

    # --- D. Post-processing (TASCAN + innovations) ---
    print("Running Innovative Post-processing...")

    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(
        1 - seg_foreground_comp,
        kernel_size=how_close_are_the_super_vox_to_boundary + 1, device=device
    )
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(
        seg_foreground_erosion, connectivity=min_touching_area
    )
    cluster_super_vox = Cluster_Super_Vox(
        min_touching_area=min_touching_area, min_touching_percentage=min_touching_percentage
    )
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

    # Innovation 1
    print(f"  - Fragment Absorption (Thresh={absorb_size_threshold})...")
    seg_foreground_absorbed = absorb_small_fragments(
        seg_foreground_single_cell_with_boundary,
        min_size_threshold=absorb_size_threshold
    )

    seg_foreground_clean = delete_too_small_cluster(
        seg_foreground_absorbed, threshold=min_cell_size_threshold
    )

    seg_final_temp = assign_boudary_voxels_to_cells_with_watershed(
        seg_foreground_clean, seg_boundary_comp, seg_background_comp, compactness=1
    )

    # Innovation 2
    print("  - Black Hole Filling...")
    seg_final = fill_black_holes(
        seg_final_temp, max_hole_size=1000, dominance_threshold=0.50
    )
    print(f"Inference Done. Time: {time.time() - start_time:.2f}s")

    # ==========================================
    #      E. Visualization (smart slice selection and correction)
    # ==========================================
    print("Generating Visualization...")

    # 1. Determine the axis
    if VISUALIZE_AXIS == 0:
        axis_name = "Z-Axis (Axial)"
        voxel_counts = np.sum(hand_seg > 0, axis=(1, 2))
    elif VISUALIZE_AXIS == 1:
        axis_name = "Y-Axis (Coronal)"
        voxel_counts = np.sum(hand_seg > 0, axis=(0, 2))
    elif VISUALIZE_AXIS == 2:
        axis_name = "X-Axis (Sagittal)"
        voxel_counts = np.sum(hand_seg > 0, axis=(0, 1))

    # 2. Select the slice adaptively
    best_slice_idx = np.argmax(voxel_counts)
    if VISUALIZE_SLICE_IDX is not None and 0 <= VISUALIZE_SLICE_IDX < len(voxel_counts):
        if voxel_counts[VISUALIZE_SLICE_IDX] == 0:
            print(f"⚠️ Warning: Slice {VISUALIZE_SLICE_IDX} is empty. Switching to best slice {best_slice_idx}.")
            slice_idx = best_slice_idx
        else:
            slice_idx = VISUALIZE_SLICE_IDX
    else:
        slice_idx = best_slice_idx

    print(f"Visualizing: {axis_name} | Slice {slice_idx}")

    # 3. Extract slices without rotation artifacts; transpose side views to make them upright
    if VISUALIZE_AXIS == 0:
        vis_raw = raw_img[slice_idx, :, :]
        vis_gt = hand_seg[slice_idx, :, :]
        vis_pred = seg_final[slice_idx, :, :]
    elif VISUALIZE_AXIS == 1:
        vis_raw = raw_img[:, slice_idx, :].T  # Use .T to make the view upright
        vis_gt = hand_seg[:, slice_idx, :].T
        vis_pred = seg_final[:, slice_idx, :].T
    elif VISUALIZE_AXIS == 2:
        vis_raw = raw_img[:, :, slice_idx].T
        vis_gt = hand_seg[:, :, slice_idx].T
        vis_pred = seg_final[:, :, slice_idx].T

    file_prefix = case_name.replace('.h5', '')
    save_name = os.path.join(save_img_dir, f"{file_prefix}_axis{VISUALIZE_AXIS}_slice{slice_idx}_3Panel.png")

    visualize_3_panel(vis_raw, vis_gt, vis_pred, save_name, pixel_size=PIXEL_SIZE_UM)

    # ==========================================
    #      F. Metrics Calculation (full evaluation)
    # ==========================================
    print("\n=== Calculating Full Metrics ===")

    # 1. Flat Metrics (ARI, ARE, VOI)
    gt_flat = hand_seg.astype(int).flatten()
    pred_flat = seg_final.astype(int).flatten()

    ari = adjusted_rand_score(gt_flat, pred_flat)
    are = adapted_rand_error(gt_flat, pred_flat)
    voi_val = VOI(seg_final.astype(int), hand_seg.astype(int))

    print(f"ARI: {ari:.4f}")
    print(f"ARE: {are}")  # (error, precision, recall)
    print(f"VOI: {voi_val}")

    # 2. IoU and Dice with downsampling for speed
    print(f"Calculating IoU/Dice (Scale: {SCALE_FACTOR}x)...")
    output_size = (int(seg_final.shape[0] * SCALE_FACTOR),
                   int(seg_final.shape[1] * SCALE_FACTOR),
                   int(seg_final.shape[2] * SCALE_FACTOR))

    hand_seg_resized = img_3d_interpolate(hand_seg, output_size, mode='nearest')
    seg_final_resized = img_3d_interpolate(seg_final, output_size, mode='nearest')

    accuracy = IOU_and_Dice_Accuracy(hand_seg_resized, seg_final_resized)
    accuracy_record = accuracy.cal_accuracy_II()

    iou_mean = np.mean(accuracy_record[:, 1])
    dice_mean = np.mean(accuracy_record[:, 2])
    iou_70 = np.mean(accuracy_record[:, 1] > 0.7)
    dice_70 = np.mean(accuracy_record[:, 2] > 0.7)
    iou_50 = np.mean(accuracy_record[:, 1] > 0.5)
    dice_50 = np.mean(accuracy_record[:, 2] > 0.5)

    print("-" * 30)
    print(f"Avg IoU:  {iou_mean:.4f}")
    print(f"Avg Dice: {dice_mean:.4f}")
    print(f"IoU > 0.7: {iou_70 * 100:.2f}%")
    print(f"Dice > 0.7: {dice_70 * 100:.2f}%")
    print(f"IoU > 0.5: {iou_50 * 100:.2f}%")
    print(f"Dice > 0.5: {dice_50 * 100:.2f}%")
    print("-" * 30)

    print("--- Done ---")
