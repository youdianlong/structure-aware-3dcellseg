import os

source_path = "/data1/myt/dataset/PNAS/"
processed_path = "/data1/myt/dataset/ATAS_processed/"
pre_cropped_path = "/data1/myt/dataset/ATAS_processed_pre_croped"
##### Ovules #####
source_file_path = "/data1/myt/dataset/ovules/"
output_file_path = "/data1/myt/dataset/ovules_processed_thin_boundary"

# step 1
run_py_script = "python prepare_dataset/prepare_Ovules_dataset.py "+\
"--source_file_path "+source_file_path+" "+\
"--output_file_path "+output_file_path+" "+\
"--img_size_scale_factor 0.5 "+\
"--width_of_membrane 1"
print("run "+run_py_script)
os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_Ovules.py "+\
"--path "+output_file_path
print("run "+run_py_script)
os.system(run_py_script)
##### Ovules #####