import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk

from utils import generate_click_prompt, random_box, random_click
from scipy.ndimage import zoom
def resize_depth(image, target_depth):
    """
    Adjust the depth of a 3D image to the target depth by cropping or padding.
    """
    current_depth = image.shape[-1]
    if current_depth == target_depth:
        return image
    elif current_depth > target_depth:
        # Crop the depth dimension
        start = (current_depth - target_depth) // 2
        return image[..., start:start + target_depth]
    else:
        # Pad the depth dimension with zeros
        pad_width = (0, 0), (0, 0), (0, target_depth - current_depth)
        return np.pad(image, pad_width, mode='constant')

def resize_image_and_mask(img, mask, target_depth, target_img_size, target_mask_size):
    """
    Resize both image and mask to desired depth and spatial size.
    """
    # Adjust depth
    img = resize_depth(img, target_depth)
    mask = resize_depth(mask, target_depth)

    # Resize spatial dimensions
    img = zoom(img, (target_img_size / img.shape[0], target_img_size / img.shape[1], 1), order=1)
    mask = zoom(mask, (target_mask_size / mask.shape[0], target_mask_size / mask.shape[1], 1), order=0)

    return img, mask
    
class ToothFairy(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        self.args = args
        self.data_path = os.path.join(data_path,'Dataset')
        #self.name_list = os.listdir('C:/Users/geyan/Projet_IAV/Medical-SAM-Adapter-main/data/Dataset/images')
        self.name_list = [f.replace('.mha', '') for f in os.listdir('C:/Users/geyan/Projet_IAV/Medical-SAM-Adapter-main/data/Dataset/images') if f.endswith('.mha')]

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        point_label = 1


        """Get the images"""
        name = self.name_list[index]
        # 加载图像和标签文件 (MHA)
        img_path = os.path.join(self.data_path, 'images', f'{name}.mha')
        #mask_path = os.path.join(self.data_path, 'labels', f'{name}.mha')
        label_name = '_'.join(name.split('_')[:-1])  # 提取如 'ToothFairy2F_063'
        mask_path = os.path.join(self.data_path, 'labels', f'{label_name}.mha')
    
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"未找到图像文件: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"未找到标签文件: {mask_path}")
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int64)
        if img.shape[0] == mask.shape[0]:
            img = np.transpose(img, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0))
        #print(f"加载图像形状: {img.shape}, 标签形状: {mask.shape}")
        # 使用 SimpleITK 加载数据
        #img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        #mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int64)
        # Adjust depth and resize
        target_depth = 64  # Example target depth
        img, mask = resize_image_and_mask(
            img, mask,
            target_depth=target_depth,
            target_img_size=self.args.image_size,
            target_mask_size=self.args.out_size
        )

        # Convert to PyTorch tensors
        img = torch.tensor(img).unsqueeze(0)  # Add channel dimension
        mask = torch.tensor(mask).unsqueeze(0).int()  # Add channel dimension
        print(f"加载图像形状: {img.shape}, 标签形状: {mask.shape}")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])
        

        # img = np.resize(mask,(self.args.image_size, self.args.image_size,img.shape[-1]))
        # mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))

        mask = torch.clamp(mask,min=0,max=1).int()

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)
        # if self.transform:
        #     state = torch.get_rng_state()
        #     img = self.transform(img)
        #     torch.set_rng_state(state)

        #     if self.transform_msk:
        #         mask = self.transform_msk(mask)
                
        #     # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
        #     #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

