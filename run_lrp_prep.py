import os

##### LRP #####
source_file_path = "/data1/myt/dataset/LateralRootPrimordia/"
output_file_path = "/data1/myt/dataset/LateralRootPrimordia_processed_wide_boundary"
output_file_path_II = "/data1/myt/dataset/LateralRootPrimordia_pre_croped"

# step 1
run_py_script = "python prepare_dataset/prepare_LateralRootPrimordia_dataset.py "+\
"--source_file_path "+source_file_path+" "+\
"--output_file_path "+output_file_path+" "+\
"--img_size_scale_factor 0.5 "+\
"--width_of_membrane 2.5"
print("run "+run_py_script)
os.system(run_py_script)

# # or you can precrop the dataset
# # run_py_script = "python prepare_dataset/pre_crop_LateralRootPrimordia.py "+\
# # "--source_file_path "+source_file_path+" "+\
# # "--output_file_path "+output_file_path_II
# # print("run "+run_py_script)
# # os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_LateralRootPrimordia.py "+\
"--path "+output_file_path
print("run "+run_py_script)
os.system(run_py_script)
##### LRP #####