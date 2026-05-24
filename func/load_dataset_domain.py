import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
import h5py

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio

from .load_dataset import Random3DCrop_np, Normalization_np



class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size,
                          (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size = output_size

    def random_crop_start_point(self, input_size):
        assert len(input_size) == 3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w = input_size
        d_new, h_new, w_new = self.output_size
        assert (
                    d >= d_new and h >= h_new and w >= w_new), "Attention: input size should >= crop size; input size: " + str(
            input_size)

        d, h, w = input_size
        d_new, h_new, w_new = self.output_size

        d_start = np.random.randint(0, d - d_new + 1)
        h_start = np.random.randint(0, h - h_new + 1)
        w_start = np.random.randint(0, w - w_new + 1)

        return d_start, h_start, w_start

    def __call__(self, img_3d, start_points=None):
        img_3d = np.array(img_3d)

        d, h, w = img_3d.shape
        d_new, h_new, w_new = self.output_size

        assert (d >= d_new and h >= h_new and w >= w_new), "Attention: input size should >= crop size"

        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)

        d_start, h_start, w_start = start_points

        crop = img_3d[d_start:d_start + d_new, h_start:h_start + h_new, w_start:w_start + w_new]

        return crop


class Normalization_np(object):
    def __init__(self):
        self.name = 'ManualNormalization'

    def __call__(self, img_3d):
        img_3d -= np.min(img_3d)
        max_99_val = np.percentile(img_3d, 99)
        if max_99_val > 0:
            img_3d = img_3d / max_99_val * 255
        return img_3d
class DomainGENDataset(Dataset):

    def __init__(self, data_dict1,data_dict2):
        # each item of data_dict is {name:{"raw":raw img path, "background": background img path,
        # "boundary": boundary img path, "foreground": foreground img path}}
        self.data_dict1 = data_dict1
        self.data_dict2 = data_dict2
        self.name_list1 = np.array(list(data_dict1))
        self.name_list2 = np.array(list(data_dict2))
        self.para = {}

    def __len__(self):
        return len(self.name_list1) + len(self.name_list2)

    def __getitem__(self, idx):
        return self.get(idx,
                        file_format1=self.para["file_format1"],
                        file_format2=self.para["file_format2"],
                        crop_size=self.para["crop_size"],
                        boundary_importance=self.para["boundary_importance"],
                        need_tensor_output=self.para["need_tensor_output"],
                        need_transform=self.para["need_transform"])

    def set_para(self, file_format1 = '.h5',file_format2 = '.npz',crop_size = (64, 64, 64),
                 boundary_importance = 1, need_tensor_output = True, need_transform = True):
        self.para["file_format1"] = file_format1
        self.para["file_format2"] = file_format2
        self.para["crop_size"] = crop_size
        self.para["boundary_importance"] = boundary_importance
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    def set_random_crop_size(self, crop_size_range=[32, 64]):
        return np.random.randint(crop_size_range[0], crop_size_range[1], size=(3))

    def get(self, idx, file_format1='.h5',file_format2='.npz', crop_size=(64, 64, 64), boundary_importance=1, need_tensor_output=True,need_transform=True):

        crop_size = tuple(crop_size)
        # print("random crop size: "+str(crop_size))
        random3dcrop = Random3DCrop_np(crop_size)

        normalization = Normalization_np()

        if idx < len(self.name_list1):
            name = self.name_list1[idx]
            if file_format1 == ".npy":
                raw_3d_img = np.load(self.data_dict1[name]["raw"])
                seg_boundary = np.load(self.data_dict1[name]["boundary"])
                seg_foreground = np.load(self.data_dict1[name]["foreground"])
                domain_label = 0
                # seg_background = np.load(self.data_dict[name]["background"])
            elif file_format1 == ".tif":
                raw_3d_img = io.imread(self.data_dict1[name]["raw"])
                seg_boundary = io.imread(self.data_dict1[name]["boundary"])
                seg_foreground = io.imread(self.data_dict1[name]["foreground"])
                domain_label = 0
                # seg_background = io.imread(self.data_dict[name]["background"])
            elif file_format1 == ".h5":
                hf = h5py.File(self.data_dict1[name], 'r')
                raw_3d_img = np.array(hf["raw"])
                seg_boundary = np.array(hf["boundary"])
                seg_foreground = np.array(hf["foreground"])
                domain_label = 0
                hf.close()
            elif file_format1 == ".npz":
                npz_file = np.load(self.data_dict1[name])
                raw_3d_img = np.array(npz_file["raw"])
                seg_boundary = np.array(npz_file["boundary"])
                seg_foreground = np.array(npz_file["foreground"])
                domain_label = 0

        elif len(self.name_list1) <= idx < len(self.name_list2) + len(self.name_list1):
            name = self.name_list2[idx - len(self.name_list1)]
            if file_format2 == ".npy":
                raw_3d_img = np.load(self.data_dict2[name]["raw"])
                seg_boundary = np.load(self.data_dict2[name]["boundary"])
                seg_foreground = np.load(self.data_dict2[name]["foreground"])
                domain_label = 1
                # seg_background = np.load(self.data_dict[name]["background"])
            elif file_format2 == ".tif":
                raw_3d_img = io.imread(self.data_dict2[name]["raw"])
                seg_boundary = io.imread(self.data_dict2[name]["boundary"])
                seg_foreground = io.imread(self.data_dict2[name]["foreground"])
                domain_label = 1
                # seg_background = io.imread(self.data_dict[name]["background"])
            elif file_format2 == ".h5":
                hf = h5py.File(self.data_dict2[name], 'r')
                raw_3d_img = np.array(hf["raw"])
                seg_boundary = np.array(hf["boundary"])
                seg_foreground = np.array(hf["foreground"])
                domain_label = 1
                hf.close()
            elif file_format2 == ".npz":
                npz_file = np.load(self.data_dict2[name])
                raw_3d_img = np.array(npz_file["raw"])
                seg_boundary = np.array(npz_file["boundary"])
                seg_foreground = np.array(npz_file["foreground"])
                domain_label = 1

        # Clean data and generate the background mask
        raw_3d_img = np.nan_to_num(raw_3d_img, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_boundary = np.nan_to_num(seg_boundary, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_foreground = np.nan_to_num(seg_foreground, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        seg_background = np.array((seg_boundary + seg_foreground) == 0, dtype=int)

        # raw_3d_img=normalization(raw_3d_img)

        # Convert data types and validate shapes
        raw_3d_img = np.array(raw_3d_img, float)
        seg_background = np.array(seg_background, float)
        seg_boundary = np.array(seg_boundary, float)
        seg_foreground = np.array(seg_foreground, float)
        assert raw_3d_img.shape == seg_background.shape
        assert seg_background.shape == seg_boundary.shape
        assert seg_boundary.shape == seg_foreground.shape

        # Random crop
        start_points = random3dcrop.random_crop_start_point(raw_3d_img.shape)
        raw_3d_img = random3dcrop(raw_3d_img, start_points=start_points)
        seg_background = random3dcrop(seg_background, start_points=start_points)
        seg_boundary = random3dcrop(seg_boundary, start_points=start_points)
        seg_foreground = random3dcrop(seg_foreground, start_points=start_points)

        # Add the channel dimension
        raw_3d_img = np.expand_dims(raw_3d_img, axis=0)
        seg_background = np.expand_dims(seg_background, axis=0)
        seg_boundary = np.expand_dims(seg_boundary, axis=0)
        seg_foreground = np.expand_dims(seg_foreground, axis=0)

        output = {'raw': raw_3d_img, 'background': seg_background, 'boundary': seg_boundary,
                  'foreground': seg_foreground,'domain_label': domain_label}

        output.update(self.get_weights(output, boundary_importance))

        if need_tensor_output:
            output = self.to_tensor(output)

            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)

        return output

    def get_weights(self, images, boundary_importance):  # images: a dict, each item should be in numpy.array format
        seg_background = images['background']
        seg_boundary = images[
                           'boundary'] * boundary_importance  # boundary is boundary_importance times more important than others
        seg_foreground = images['foreground']

        seg_background_zeros = np.array(seg_background == 0, dtype=int) * 0.5
        seg_boundary_zeros = np.array(seg_boundary == 0, dtype=int) * 0.5
        seg_foreground_zeros = np.array(seg_foreground == 0, dtype=int) * 0.5

        return {'weights_background': seg_background + seg_background_zeros,
                'weights_boundary': seg_boundary + seg_boundary_zeros,
                'weights_foreground': seg_foreground + seg_foreground_zeros}

    def to_tensor(self, images):
        images_tensor={}
        for item in images.keys():
            if item == 'domain_label':
                images_tensor[item] = images[item]
            else:
                images_tensor[item] = from_numpy(images[item].copy()).float()
        return images_tensor

    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            if item == 'domain_label':
                continue 
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob)])
        subject_all_imgs = transform_shape(subject_all_imgs)
        
        if 'raw' in subject_all_imgs:
            transform_val = tio.Compose([
                tio.RandomBlur(p=prob), tio.RandomNoise(p=prob), tio.RandomMotion(p=prob),
                tio.RandomBiasField(p=prob), tio.RandomSpike(p=prob), tio.RandomGhosting(p=prob)])
            subject_all_imgs['raw'] = transform_val(subject_all_imgs['raw'])
        
        output_tensors = {}
        for item in subject_all_imgs.keys():
            output_tensors[item] = subject_all_imgs[item].data
        output_tensors['domain_label'] = image_tensors['domain_label']
        
        return output_tensors

if __name__ == '__main__':
    import pickle
    import os
    from torch.utils.data import DataLoader

    ATAS_BASE_PATH = '/data1/myt/dataset/ATAS_processed/'
    OVULES_BASE_PATH = '/data1/myt/dataset/ovules_processed_thin_boundary/train'
    
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    print("--- 正在测试 DomainGENDataset ---")

    # --- 1. Load two pkl files with different formats ---
    print("加载 pkl...")

    # Domain 0: HMS (uses .h5 and structure A)
    ATAS_dataset_info = load_obj('dataset_info/ATAS_dataset_info')
    ATAS_data_dict = {f'Ovules_{i}': os.path.join(ATAS_BASE_PATH,path) for i, path in enumerate(ATAS_dataset_info['train'])}  # Already in {name: {"raw":...}} format
    print(f"域 0 (ATAS) 加载了 {len(ATAS_data_dict)} 个样本")

    # Domain 1: LRP (uses .npz and structure B)
    Ovules_dataset_info = load_obj('dataset_info/Ovules_dataset_info')

    # Key conversion: the Ovules 'train' split is a list.
    # Convert it into a structure-B dictionary ({name: path}) that get() can read.
    Ovules_data_dict = {f'Ovules_{i}': os.path.join(OVULES_BASE_PATH,path) for i, path in enumerate(Ovules_dataset_info['train'])}
    print(f"域 1 (LRP) 加载了 {len(Ovules_data_dict)} 个样本")

    # --- 2. Instantiate the Dataset ---
    dataset = DomainGENDataset(data_dict1=ATAS_data_dict, data_dict2=Ovules_data_dict)
    print(f"Dataset 初始化成功，总长度: {len(dataset)}")

    # --- 3. Set parameters ---
    dataset.set_para(
        file_format1='.h5',  # Set the file format for domain 1
        file_format2='.npz',  # Set the file format for domain 2
        crop_size=(64, 64, 64),
        boundary_importance=1,
        need_tensor_output=True,
        need_transform=True  # Enable data augmentation testing
    )
    print("参数设置成功。")

    # --- 4. Create the DataLoader ---
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,  # Use a slightly larger batch size
        shuffle=True,  # Must be True to mix domains
        num_workers=0  # Must be 0 during testing
    )

    # --- 5. Fetch one batch and check it ---
    print("正在从 DataLoader 获取一个混合批次...")
    try:
        batch = next(iter(loader))

        print("\n--- 批次获取成功 ---")
        print("批次中包含的键:")
        for key in batch.keys():
            print(f"  {key}: {batch[key].shape}")

        print("\n--- 关键检查 ---")
        domain_labels = batch['domain_label']
        print(f"  域标签张量: {domain_labels}")

        # Check whether the domain labels are mixed
        unique_labels = torch.unique(domain_labels)
        if len(unique_labels) > 1:
            print(" 验证成功：批次中包含了混合的域标签 (0 和 1)！")
        else:
            print(f" 验证注意：批次中只包含一个域 ({unique_labels})。")
            print("   (如果 batch_size 较小或偶然发生，这没问题。请多运行几次。)")

    except Exception as e:
        print(f"\n 从 DataLoader 获取批次时失败: {e}")
        print("   请仔细检查您的 pkl 路径和 get() 方法中的加载逻辑。")
        raise e

