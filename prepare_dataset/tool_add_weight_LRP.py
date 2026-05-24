# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import tqdm

# ================= CONFIGURATION =================
# 1. Source Directory (Your current LRP data)
source_dir = "/data1/myt/dataset/LateralRootPrimordia_processed_wide_boundary/"

# 2. Target Directory (Will be created automatically)
target_dir = "/data1/myt/dataset/LRP_processed_with_weights_final"
# =================================================

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("Status: Start processing...")
print("Source: " + source_dir)
print("Target: " + target_dir)

# Collect all h5 files
files_to_process = []
for root, dirs, files in os.walk(source_dir):
    for file_name in files:
        if file_name.endswith('.h5'):
            files_to_process.append((root, file_name))

print("Found " + str(len(files_to_process)) + " files.")

# Process files
for src_root, file_name in tqdm.tqdm(files_to_process, desc="Generating"):
    # Construct paths
    src_path = os.path.join(src_root, file_name)
    
    # Maintain directory structure
    relative_path = os.path.relpath(src_root, source_dir)
    dst_folder = os.path.join(target_dir, relative_path)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    dst_path = os.path.join(dst_folder, file_name)

    # Open Source (Read) and Target (Write)
    with h5py.File(src_path, 'r') as f_src, h5py.File(dst_path, 'w') as f_dst:
        
        # --- 1. Copy Original Data ---
        for key in f_src.keys():
            f_src.copy(key, f_dst)
        
        # --- 2. Check and Generate Weights ---
        if 'weights_boundary' in f_dst:
            # Skip if weights already exist
            continue
            
        try:
            # Read masks
            background = f_dst['background'][:]
            boundary = f_dst['boundary'][:]
            foreground = f_dst['foreground'][:]
            
            # --- 3. Calculate Weights ---
            # Weight for Background = 1.0
            w_bg = np.ones_like(background, dtype=np.float32)
            
            # Weight for Boundary = 4.0 (where boundary > 0), else 1.0
            w_bd = np.ones_like(boundary, dtype=np.float32)
            w_bd[boundary > 0] = 4.0 
            
            # Weight for Foreground = 1.0
            w_fg = np.ones_like(foreground, dtype=np.float32)
            
            # --- 4. Write Weights to New File ---
            # Using gzip compression to save space
            f_dst.create_dataset('weights_background', data=w_bg, compression="gzip")
            f_dst.create_dataset('weights_boundary', data=w_bd, compression="gzip")
            f_dst.create_dataset('weights_foreground', data=w_fg, compression="gzip")
            
        except KeyError as e:
            print("\nSkipping " + file_name + ": Missing mask data (" + str(e) + ")")
        except Exception as e:
            print("\nError processing " + file_name + ": " + str(e))

print("\nStatus: All Done!")
print("Please update your training script to use the new path:")
print(target_dir)