# toothfairy.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import generate_click_prompt, random_click
import torch.nn.functional as F
from collections import defaultdict

class ToothFairy(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', plane=False):
        self.args = args
        self.data_path = os.path.join(data_path, 'Dataset')
        self.img_dir = os.path.join(self.data_path, 'images')
        self.label_dir = os.path.join(self.data_path, 'labels')

        self.name_list = [f.replace('_data.npy', '') for f in os.listdir(self.img_dir) if f.endswith('_data.npy')]

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.num_classes = 49  

        self.patient_to_indices = self._group_indices_by_patient()

    def _group_indices_by_patient(self):
        """
        Group the data indexes based on patient IDs.
        Assume that each name in the name_list is in the format 'patientID_slice_XX'.
        """
        patient_dict = defaultdict(list)
        for idx, name in enumerate(self.name_list):
            patient_id = name.split('_slice_')[0]
            patient_dict[patient_id].append(idx)
        return patient_dict

    def get_patient_ids(self):
        """
        Returns a list of all patient IDs.
        """
        return list(self.patient_to_indices.keys())

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
      
        if isinstance(index, list):  # Batch Indexing
            data_batch = [self.__getitem__(i) for i in index]
            return [item for sublist in data_batch for item in (sublist if isinstance(sublist, list) else [sublist])]
        
            
        point_label = 1  # Default click points are positive samples

        name = self.name_list[index]
        img_path = os.path.join(self.img_dir, f'{name}_data.npy')
        mask_path = os.path.join(self.label_dir, f'{name}_gt_sparse.npy')

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or label file not found: {img_path}, {mask_path}")

        img = np.load(img_path).astype(np.float32)  
        mask = np.load(mask_path).astype(np.int64)  
        
        img = torch.tensor(img).unsqueeze(0)
        img = img.repeat(3, 1, 1)  

        # Convert mask to one-hot format
        mask = torch.tensor(mask).unsqueeze(0)  
        mask = F.one_hot(mask, num_classes=self.num_classes).permute(0, 3, 1, 2).squeeze(0)
        #  mask Final shape as [num_classes, H, W]

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)
            pt = torch.tensor([[pt[2], pt[1]]])  # Correct the order by adding a click point count dimension with a depth of 0

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt, 
            'image_meta_dict': {'filename_or_obj': name},
        }
