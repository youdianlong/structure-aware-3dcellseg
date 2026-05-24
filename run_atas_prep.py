import os

##### ATAS #####
# please change if needed
source_path = "/data1/myt/dataset/PNAS/"
processed_path = "/data1/myt/dataset/ATAS_processed/"
pre_cropped_path = "/data1/myt/dataset/ATAS_processed_pre_croped"

# step 1
run_py_script = "python prepare_dataset/prepare_ATAS_dataset.py "+\
"--source_file_path "+source_path+" "+\
"--output_file_path "+processed_path+" "+\
"--width_of_membrane 1.5"
print("run "+run_py_script)
os.system(run_py_script)

# step 2
run_py_script = "python prepare_dataset/get_the_dataset_info_of_ATAS.py "+\
"--path "+processed_path+" "+\
"--test_name 'plant15' "
print("run "+run_py_script)
os.system(run_py_script)

# step 3
# use the dataset_info generated from get_the_dataset_info_of_ATAS.py
run_py_script = "python prepare_dataset/pre_crop_ATAS.py "+\
"--output_path "+pre_cropped_path+" "
print("run "+run_py_script)
os.system(run_py_script)

# step 4
run_py_script = "python prepare_dataset/get_the_dataset_info_of_ATAS_pre_cropped.py "+\
"--path "+pre_cropped_path
print("run "+run_py_script)
os.system(run_py_script)
##### ATAS #####